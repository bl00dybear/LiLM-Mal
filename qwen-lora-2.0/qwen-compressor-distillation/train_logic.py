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
):
    strategy = getattr(config, "strategy", "fsdp")

    # doar trainabilele (LoRA_E + memory + ae) — checkpoint de ~160MB, nu 3GB
    state_dict = get_trainable_state_dict(model, strategy=strategy)

    if rank != 0:
        return

    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        "trainable_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    if step is not None:
        fname = f"compressor_ep{epoch}_step{step}.pt"
    else:
        fname = f"compressor_ep{epoch}.pt"

    save_path = os.path.join(config.output_dir, fname)
    torch.save(checkpoint, save_path)
    print(f"[info] [rank 0] checkpoint salvat: {save_path}")

    if step is not None:
        pattern = os.path.join(config.output_dir, "compressor_ep*_step*.pt")
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


def get_no_sync_context(model, strategy="fsdp"):
    if strategy == "ddp":
        return model.no_sync()
    # FSDP2 + replicate: sync la fiecare pas, overhead neglijabil pe LoRA
    return nullcontext()


def evaluate(model, loader, rank):
    model.eval()

    # sume ponderate cu batch size, reduse global:
    # [loss, rec, logit, repr, cos, agree_teacher, acc_label, count]
    sums = torch.zeros(8, device=rank, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            code_ids  = batch["code_ids"].to(rank)
            code_mask = batch["code_mask"].to(rank)
            h_t       = batch["h_t"].to(rank)
            z_t       = batch["z_t"].to(rank)
            labels    = batch["labels"].to(rank)

            out = model(code_ids=code_ids, code_mask=code_mask, h_t=h_t, z_t=z_t)

            bsz = code_ids.size(0)
            z_s = out["z_s"]
            sums[0] += out["loss"].item() * bsz
            sums[1] += out["loss_rec"].item() * bsz
            sums[2] += out["loss_logit"].item() * bsz
            sums[3] += out["loss_repr"].item() * bsz
            sums[4] += out["cos"].item() * bsz
            sums[5] += ((z_s > 0) == (z_t > 0)).float().sum().item()
            sums[6] += ((z_s > 0).long() == labels).float().sum().item()
            sums[7] += bsz

    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    count = max(sums[7].item(), 1.0)

    return {
        "val_loss":       sums[0].item() / count,
        "val_loss_rec":   sums[1].item() / count,
        "val_loss_logit": sums[2].item() / count,
        "val_loss_repr":  sums[3].item() / count,
        "val_cos":        sums[4].item() / count,
        "val_agree":      sums[5].item() / count,
        "val_acc":        sums[6].item() / count,
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
) -> None:
    is_distributed = dist.is_available() and dist.is_initialized()
    strategy = getattr(config, "strategy", "fsdp")
    global_step = max(0, int(start_global_step))
    resume_micro_batches = global_step * grad_accum_steps

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_memory_tokens = int(config.num_memory_tokens)

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
                        f"[info] resume activ: skip {resume_micro_batches} batches locale "
                        f"({samples_to_skip} segmente/rank)"
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
            # tokeni reali procesati de encoder (cod + memory), numarati pe CPU
            bsz = batch["code_ids"].size(0)
            window_tokens += int(batch["code_mask"].sum()) + bsz * num_memory_tokens

            code_ids  = batch["code_ids"].to(rank)
            code_mask = batch["code_mask"].to(rank)
            h_t       = batch["h_t"].to(rank)
            z_t       = batch["z_t"].to(rank)

            is_sync_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))

            sync_context = nullcontext() if is_sync_step else get_no_sync_context(model, strategy)

            with sync_context:
                out = model(code_ids=code_ids, code_mask=code_mask, h_t=h_t, z_t=z_t)
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
                        "train/loss_rec":              out["loss_rec"].item(),
                        "train/loss_logit":            out["loss_logit"].item(),
                        "train/loss_repr":             out["loss_repr"].item(),
                        "train/cos":                   out["cos"].item(),
                        "train/lr":                    scheduler.get_last_lr()[0],
                        "train/global_step":           global_step,
                        "train/vram_peak_alloc_gb":    torch.cuda.max_memory_allocated(rank) / 1024**3,
                        "train/vram_peak_reserved_gb": torch.cuda.max_memory_reserved(rank) / 1024**3,
                        "train/grad_norm":             grad_norm,
                        "train/throughput_tokens_sec": tokens_sec,
                    })
                    progress_bar.set_postfix({
                        "loss": f"{actual_loss:.4f}",
                        "cos":  f"{out['cos'].item():.3f}",
                        "tok/s": f"{tokens_sec:.0f}",
                    })

                torch.cuda.reset_peak_memory_stats(rank)
                window_tokens = 0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_start_time = time.time()

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
                    )

                if global_step % config.evaluate_every_n_steps == 0:
                    dist.barrier()
                    metrics = evaluate(model, valid_loader, rank)

                    if rank == 0:
                        print(
                            f"\n[info] Step {global_step} | "
                            f"Val Loss: {metrics['val_loss']:.4f} | "
                            f"Cos: {metrics['val_cos']:.4f} | "
                            f"Agree: {metrics['val_agree']:.4f} | "
                            f"Acc: {metrics['val_acc']:.4f}"
                        )
                        wandb.log({
                            "val/loss":       metrics["val_loss"],
                            "val/loss_rec":   metrics["val_loss_rec"],
                            "val/loss_logit": metrics["val_loss_logit"],
                            "val/loss_repr":  metrics["val_loss_repr"],
                            "val/cos":        metrics["val_cos"],
                            "val/agree":      metrics["val_agree"],
                            "val/acc":        metrics["val_acc"],
                            "train/global_step": global_step,
                        })

                    model.train()

        dist.barrier()
        metrics = evaluate(model, valid_loader, rank)

        if rank == 0:
            print(
                f"\n[info] Epoch {epoch} finished | "
                f"Val Loss: {metrics['val_loss']:.4f} | "
                f"Cos: {metrics['val_cos']:.4f} | "
                f"Agree: {metrics['val_agree']:.4f}"
            )
            wandb.log({
                "val/loss":       metrics["val_loss"],
                "val/loss_rec":   metrics["val_loss_rec"],
                "val/loss_logit": metrics["val_loss_logit"],
                "val/loss_repr":  metrics["val_loss_repr"],
                "val/cos":        metrics["val_cos"],
                "val/agree":      metrics["val_agree"],
                "val/acc":        metrics["val_acc"],
                "epoch":          epoch,
            })

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )
