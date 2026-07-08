import os
import glob
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm.auto import tqdm
from contextlib import nullcontext
import wandb

from utils import get_state_dict_for_save


def save_model(
    model,
    optimizer,
    scheduler,
    rank,
    config,
    epoch,
    global_step,
    step=None,
):
    strategy = getattr(config, "strategy", "ddp")

    # FSDP: get_state_dict_for_save is a collective — ALL ranks must call it
    state_dict = get_state_dict_for_save(model, strategy=strategy)

    if rank != 0:
        return

    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    if step is not None:
        fname = f"qwen_malware_ep{epoch}_step{step}.pt"
    else:
        fname = f"qwen_malware_ep{epoch}.pt"

    save_path = os.path.join(config.output_dir, fname)
    torch.save(checkpoint, save_path)
    print(f"[info] [rank 0] checkpoint salvat: {save_path}")

    if step is not None:
        pattern = os.path.join(config.output_dir, "qwen_malware_ep*_step*.pt")
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


def get_gradient_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5


def get_no_sync_context(model, strategy="ddp"):
    """Return a context manager that disables gradient sync for non-sync steps."""
    if strategy == "ddp":
        return model.no_sync()
    # FSDP2 + replicate: no_sync not directly available.
    # With 2 GPUs and small LoRA params, the overhead of syncing every step
    # is negligible, so we just return nullcontext.
    return nullcontext()


def evaluate(model, loader, rank):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels         = batch["labels"].to(rank)
            chunk_mask     = batch["chunk_mask"].to(rank) if "chunk_mask" in batch else None

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, chunk_mask=chunk_mask)
            loss    = outputs["loss"]
            logits  = outputs["logits"]

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / max(len(loader), 1)

    local_preds  = torch.cat(all_preds)
    local_labels = torch.cat(all_labels)

    world_size = dist.get_world_size()

    max_len = torch.tensor([local_preds.size(0)], dtype=torch.long, device=rank)
    dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
    max_len = max_len.item()

    def pad_tensor(t):
        pad_size = max_len - t.size(0)
        if pad_size > 0:
            pad = torch.full((pad_size,), -1, dtype=t.dtype, device=t.device)
            return torch.cat([t, pad])
        return t

    local_preds_padded  = pad_tensor(local_preds).to(rank)
    local_labels_padded = pad_tensor(local_labels).to(rank)

    gathered_preds  = [torch.zeros_like(local_preds_padded)  for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(local_labels_padded) for _ in range(world_size)]

    dist.all_gather(gathered_preds,  local_preds_padded)
    dist.all_gather(gathered_labels, local_labels_padded)

    loss_tensor = torch.tensor([avg_loss], device=rank)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / world_size

    metrics = {"val_loss": global_avg_loss}

    if rank == 0:
        global_preds  = torch.cat(gathered_preds).cpu()
        global_labels = torch.cat(gathered_labels).cpu()

        valid_mask    = global_labels != -1
        global_preds  = global_preds[valid_mask].numpy()
        global_labels = global_labels[valid_mask].numpy()

        metrics["accuracy"]  = accuracy_score(global_labels, global_preds)
        metrics["precision"] = precision_score(global_labels, global_preds, zero_division=0)
        metrics["recall"]    = recall_score(global_labels, global_preds, zero_division=0)
        metrics["f1"]        = f1_score(global_labels, global_preds, zero_division=0)

    return metrics


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    grad_accum_steps: int,
    rank: int,
    config,
    start_global_step: int = 0,
) -> None:
    is_distributed = dist.is_available() and dist.is_initialized()
    strategy = getattr(config, "strategy", "ddp")
    global_step = max(0, int(start_global_step))
    resume_micro_batches = global_step * grad_accum_steps

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
                        f"({samples_to_skip} samples/rank)"
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
            # real (non-padded-chunk) tokens in this micro-batch, counted on CPU
            if "chunk_mask" in batch:
                window_tokens += int(batch["chunk_mask"].sum()) * batch["input_ids"].size(-1)
            else:
                window_tokens += batch["input_ids"].size(0) * batch["input_ids"].size(1) * batch["input_ids"].size(2)

            input_ids      = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels         = batch["labels"].to(rank)
            chunk_mask     = batch["chunk_mask"].to(rank) if "chunk_mask" in batch else None
            batch_size     = input_ids.size(0)

            is_sync_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))

            sync_context = nullcontext() if is_sync_step else get_no_sync_context(model, strategy)

            with sync_context:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, chunk_mask=chunk_mask)
                loss    = outputs["loss"] / grad_accum_steps
                loss.backward()

            if is_sync_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

                if rank == 0:
                    actual_loss = loss.item() * grad_accum_steps
                    step_time   = time.time() - step_start_time
                    throughput  = (batch_size * grad_accum_steps * dist.get_world_size()) / step_time if step_time > 0 else 0
                    tokens_sec  = (window_tokens * dist.get_world_size()) / step_time if step_time > 0 else 0

                    wandb.log({
                        "train/loss":                  actual_loss,
                        "train/lr":                    scheduler.get_last_lr()[0],
                        "train/global_step":           global_step,
                        "train/vram_peak_alloc_gb":    torch.cuda.max_memory_allocated(rank) / 1024**3,
                        "train/vram_peak_reserved_gb": torch.cuda.max_memory_reserved(rank) / 1024**3,
                        "train/grad_norm":             grad_norm,
                        "train/throughput_samples_sec": throughput,
                        "train/throughput_tokens_sec": tokens_sec,
                    })
                    progress_bar.set_postfix({
                        "loss":      f"{actual_loss:.4f}",
                        "tok/s":     f"{tokens_sec:.0f}",
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
                    save_model(
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
                            f"F1: {metrics['f1']:.4f} | "
                            f"Recall: {metrics['recall']:.4f}"
                        )
                        wandb.log({
                            "val/loss":      metrics["val_loss"],
                            "val/accuracy":  metrics["accuracy"],
                            "val/precision": metrics["precision"],
                            "val/recall":    metrics["recall"],
                            "val/f1":        metrics["f1"],
                            "train/global_step": global_step,
                        })

                    model.train()

        dist.barrier()
        metrics = evaluate(model, valid_loader, rank)

        if rank == 0:
            print(
                f"\n[info] Epoch {epoch} finished | "
                f"Val Loss: {metrics['val_loss']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )
            wandb.log({
                "val/loss":      metrics["val_loss"],
                "val/accuracy":  metrics["accuracy"],
                "val/precision": metrics["precision"],
                "val/recall":    metrics["recall"],
                "val/f1":        metrics["f1"],
                "epoch":         epoch,
            })

        # End-of-epoch checkpoint (all ranks participate for FSDP)
        save_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )