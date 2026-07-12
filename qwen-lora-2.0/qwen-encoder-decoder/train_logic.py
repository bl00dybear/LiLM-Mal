import os
import glob
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tqdm.auto import tqdm
from contextlib import nullcontext
import wandb

from utils import get_trainable_state_dict


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    rank,
    config,
    epoch,
    global_step,
    step=None,
    best=False,
    early_stop_state=None,
):
    state_dict = get_trainable_state_dict(model)

    if rank != 0:
        return

    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        "trainable_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "early_stop_state": early_stop_state,
    }

    if best:
        fname = getattr(config, "best_checkpoint_name", "encdec_best.pt")
    elif step is not None:
        fname = f"encdec_ep{epoch}_step{step}.pt"
    else:
        fname = f"encdec_ep{epoch}.pt"

    save_path = os.path.join(config.output_dir, fname)
    torch.save(checkpoint, save_path)
    print(f"[info] [rank 0] checkpoint saved: {save_path}")

    if step is not None and not best:
        pattern = os.path.join(config.output_dir, "encdec_ep*_step*.pt")
        files = glob.glob(pattern)

        def extract_step(filepath):
            try:
                name = os.path.basename(filepath)
                return int(name.split("_step")[1].split(".pt")[0])
            except Exception:
                return -1

        files.sort(key=extract_step)
        while len(files) > 2:
            to_delete = files.pop(0)
            if os.path.exists(to_delete):
                os.remove(to_delete)


def evaluate(model, loader, rank):
    model.eval()

    sums = torch.zeros(6, device=rank, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            code_ids  = batch["code_ids"].to(rank)
            code_mask = batch["code_mask"].to(rank)
            labels    = batch["labels"].to(rank)

            out = model(code_ids=code_ids, code_mask=code_mask, labels=labels)

            bsz = labels.size(0)
            preds = (out["logits"] > 0).long()
            sums[0] += out["loss"].item() * bsz
            sums[1] += (preds == labels).float().sum().item()
            sums[2] += ((preds == 1) & (labels == 1)).float().sum().item()
            sums[3] += ((preds == 1) & (labels == 0)).float().sum().item()
            sums[4] += ((preds == 0) & (labels == 1)).float().sum().item()
            sums[5] += bsz

    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    count = max(sums[5].item(), 1.0)
    tp, fp, fn = sums[2].item(), sums[3].item(), sums[4].item()
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1.0)

    return {
        "val_loss": sums[0].item() / count,
        "val_acc":  sums[1].item() / count,
        "val_f1":   f1,
    }


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    grad_accum_steps: int,
    rank: int,
    config,
    start_global_step: int = 0,
    total_steps: int = None,
    early_stop_state: dict = None,
) -> None:
    is_distributed = dist.is_available() and dist.is_initialized()
    global_step = max(0, int(start_global_step))
    resume_micro_batches = global_step * grad_accum_steps

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_memory_tokens = int(config.num_memory_tokens)

    patience = int(getattr(config, "early_stop_patience", 4))
    min_delta = float(getattr(config, "early_stop_min_delta", 0.005))
    decay_fraction = float(getattr(config, "decay_fraction", 0.1))
    if total_steps is None:
        total_steps = max(1, (len(train_loader) // grad_accum_steps) * epochs)
    stable_end_step = max(1, int(total_steps * (1.0 - decay_fraction)))

    state = early_stop_state or {}
    best_val_loss = float(state.get("best_val_loss", float("inf")))
    evals_since_best = int(state.get("evals_since_best", 0))
    stop_training = False

    def on_validation(metrics: dict, epoch: int):
        nonlocal best_val_loss, evals_since_best

        improved = metrics["val_loss"] < best_val_loss * (1.0 - min_delta)
        if improved:
            best_val_loss = metrics["val_loss"]
            evals_since_best = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rank=rank,
                config=config,
                epoch=epoch,
                global_step=global_step,
                best=True,
                early_stop_state={"best_val_loss": best_val_loss, "evals_since_best": 0},
            )
            if rank == 0:
                print(f"[info] new best val_loss: {best_val_loss:.4f}")
        else:
            evals_since_best += 1
            if rank == 0:
                print(f"[info] no val_loss improvement ({evals_since_best}/{patience} evals)")

        if not scheduler.in_decay and evals_since_best >= patience:
            decay_steps = min(
                max(1, int(decay_fraction * global_step)),
                max(1, total_steps - global_step),
            )
            scheduler.begin_decay(decay_steps)
            if rank == 0:
                print(
                    f"[info] val_loss plateau at global_step {global_step} — "
                    f"starting LR decay over {decay_steps} steps, then stopping"
                )

    for epoch in range(epochs):
        model.train()

        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_start_index"):
            if epoch == 0 and resume_micro_batches > 0:
                samples_to_skip = resume_micro_batches * train_loader.batch_size
                train_loader.sampler.set_start_index(samples_to_skip)
                if rank == 0:
                    print(
                        f"[info] resume active: skipping {resume_micro_batches} local batches "
                        f"({samples_to_skip} files/rank)"
                    )
            else:
                train_loader.sampler.set_start_index(0)

        optimizer.zero_grad()
        step_start_time = time.time()
        window_tokens = 0
        torch.cuda.reset_peak_memory_stats(rank)

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            disable=(rank != 0),
        )

        for step, batch in progress_bar:
            n_seg = batch["code_ids"].size(0)
            window_tokens += int(batch["code_mask"].sum()) + n_seg * num_memory_tokens

            code_ids  = batch["code_ids"].to(rank)
            code_mask = batch["code_mask"].to(rank)
            labels    = batch["labels"].to(rank)

            is_sync_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))

            sync_context = nullcontext() if is_sync_step else model.no_sync()

            with sync_context:
                out = model(code_ids=code_ids, code_mask=code_mask, labels=labels)
                loss = out["loss"] / grad_accum_steps
                loss.backward()

            if is_sync_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()

                if rank == 0:
                    actual_loss = loss.item() * grad_accum_steps
                    step_time   = time.time() - step_start_time
                    tokens_sec  = (window_tokens * dist.get_world_size()) / step_time if step_time > 0 else 0

                    wandb.log({
                        "train/loss":                  actual_loss,
                        "train/lr":                    scheduler.get_last_lr()[0],
                        "train/global_step":           global_step,
                        "train/vram_peak_alloc_gb":    torch.cuda.max_memory_allocated(rank) / 1024**3,
                        "train/vram_peak_reserved_gb": torch.cuda.max_memory_reserved(rank) / 1024**3,
                        "train/grad_norm":             grad_norm,
                        "train/throughput_tokens_sec": tokens_sec,
                    })
                    progress_bar.set_postfix({
                        "loss": f"{actual_loss:.4f}",
                        "tok/s": f"{tokens_sec:.0f}",
                    })

                torch.cuda.reset_peak_memory_stats(rank)
                window_tokens = 0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_start_time = time.time()

                if not scheduler.in_decay and global_step >= stable_end_step:
                    decay_steps = max(1, total_steps - global_step)
                    scheduler.begin_decay(decay_steps)
                    if rank == 0:
                        print(
                            f"\n[info] step budget almost exhausted ({global_step}/{total_steps}) — "
                            f"starting LR decay over {decay_steps} steps"
                        )

                if global_step % config.save_every_n_steps == 0:
                    dist.barrier()
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        rank=rank,
                        config=config,
                        epoch=epoch,
                        global_step=global_step,
                        step=global_step,
                        early_stop_state={
                            "best_val_loss": best_val_loss,
                            "evals_since_best": evals_since_best,
                        },
                    )

                if global_step % config.evaluate_every_n_steps == 0:
                    dist.barrier()
                    metrics = evaluate(model, valid_loader, rank)
                    on_validation(metrics, epoch)

                    if rank == 0:
                        print(
                            f"\n[info] Step {global_step} | "
                            f"Val Loss: {metrics['val_loss']:.4f} | "
                            f"Acc: {metrics['val_acc']:.4f} | "
                            f"F1: {metrics['val_f1']:.4f}"
                        )
                        wandb.log({
                            "val/loss":      metrics["val_loss"],
                            "val/acc":       metrics["val_acc"],
                            "val/f1":        metrics["val_f1"],
                            "val/best_loss": best_val_loss,
                            "val/evals_since_best": evals_since_best,
                            "train/global_step": global_step,
                        })

                    model.train()

                if scheduler.decay_finished:
                    stop_training = True
                    if rank == 0:
                        print(f"\n[info] LR decay finished at global_step {global_step} — stopping training")
                    break

        dist.barrier()
        metrics = evaluate(model, valid_loader, rank)
        on_validation(metrics, epoch)

        if rank == 0:
            print(
                f"\n[info] Epoch {epoch} finished | "
                f"Val Loss: {metrics['val_loss']:.4f} | "
                f"Acc: {metrics['val_acc']:.4f} | "
                f"F1: {metrics['val_f1']:.4f}"
            )
            wandb.log({
                "val/loss":      metrics["val_loss"],
                "val/acc":       metrics["val_acc"],
                "val/f1":        metrics["val_f1"],
                "val/best_loss": best_val_loss,
                "epoch":         epoch,
            })

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
            config=config,
            epoch=epoch,
            global_step=global_step,
            early_stop_state={
                "best_val_loss": best_val_loss,
                "evals_since_best": evals_since_best,
            },
        )

        if stop_training:
            if rank == 0:
                print(
                    f"[info] early stopping: best val_loss {best_val_loss:.4f} saved to "
                    f"{getattr(config, 'best_checkpoint_name', 'encdec_best.pt')}"
                )
            break
