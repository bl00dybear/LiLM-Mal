import os
import glob
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm.auto import tqdm
from contextlib import nullcontext
import wandb



def save_fsdp_model(model, rank, config, epoch, is_best=False, step=None):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        if is_best:
            fname = config.best_checkpoint_name
        elif step is not None:
            fname = f"qwen_malware_ep{epoch}_step{step}.pt"
        else:
            fname = f"qwen_malware_ep{epoch}.pt"
            
        save_path = os.path.join(config.output_dir, fname)
        torch.save(cpu_state, save_path)

        if step is not None and not is_best:
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
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5





def evaluate(model, loader, rank):
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels = batch["labels"].to(rank)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs
                
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            total_loss += loss.item()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader)
    
    local_preds = torch.cat(all_preds)
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

    local_preds_padded = pad_tensor(local_preds).to(rank)
    local_labels_padded = pad_tensor(local_labels).to(rank)

    gathered_preds = [torch.zeros_like(local_preds_padded) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(local_labels_padded) for _ in range(world_size)]

    dist.all_gather(gathered_preds, local_preds_padded)
    dist.all_gather(gathered_labels, local_labels_padded)

    loss_tensor = torch.tensor([avg_loss], device=rank)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / world_size

    metrics = {"val_loss": global_avg_loss}

    if rank == 0:
        global_preds = torch.cat(gathered_preds)
        global_labels = torch.cat(gathered_labels)
        
        valid_mask = global_labels != -1
        global_preds = global_preds[valid_mask].cpu().numpy()
        global_labels = global_labels[valid_mask].cpu().numpy()

        metrics["accuracy"] = accuracy_score(global_labels, global_preds)
        metrics["precision"] = precision_score(global_labels, global_preds, zero_division=0)
        metrics["recall"] = recall_score(global_labels, global_preds, zero_division=0)
        metrics["f1"] = f1_score(global_labels, global_preds, zero_division=0)

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
    config
) -> None:
    is_distributed = dist.is_available() and dist.is_initialized()
    best_f1_score = -1.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        if is_distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        step_start_time = time.time()

        progress_bar = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch}", disable=is_distributed and rank != 0
        )

        for step, batch in progress_bar:
            input_ids      = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels         = batch["labels"].to(rank)
            batch_size     = input_ids.size(0)

            is_sync_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))

            sync_context = nullcontext()
            if is_distributed and not is_sync_step:
                if hasattr(model, "no_sync"):
                    sync_context = model.no_sync()

            with sync_context:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss    = outputs["loss"]
                loss    = loss / grad_accum_steps
                loss.backward()

            if is_sync_step:
                grad_norm = get_gradient_norm(model)
                model.clip_grad_norm_(1.0)
                
                if rank == 0:
                    actual_loss = loss.item() * grad_accum_steps
                    
                    step_time = time.time() - step_start_time
                    throughput = (batch_size * grad_accum_steps * dist.get_world_size()) / step_time if step_time > 0 else 0
                    
                    wandb.log({
                        "train/loss": actual_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/vram_gb": torch.cuda.memory_allocated(rank) / 1024**3,
                        "train/grad_norm": grad_norm,
                        "train/throughput_samples_sec": throughput
                    })
                    progress_bar.set_postfix({
                        "loss": f"{actual_loss:.4f}",
                        "samples/s": f"{throughput:.1f}"
                    })

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_start_time = time.time()

                if global_step % config.save_every_n_steps == 0 and global_step > 0:
                    save_fsdp_model(model, rank, config, epoch, is_best=False, step=global_step)

                if global_step % config.evaluate_every_n_steps == 0 and global_step > 0:
                    metrics = evaluate(model, valid_loader, rank)
                    is_best_step = False
                    
                    if rank == 0:
                        val_loss = metrics["val_loss"]
                        f1 = metrics["f1"]
                        print(f"\n[info] Step {global_step} eval. Val Loss: {val_loss:.4f} | F1: {f1:.4f} | Recall: {metrics['recall']:.4f}")
                        
                        wandb.log({
                            "val/loss": val_loss,
                            "val/accuracy": metrics["accuracy"],
                            "val/precision": metrics["precision"],
                            "val/recall": metrics["recall"],
                            "val/f1": f1,
                            "train/global_step": global_step
                        })
                        
                        if f1 > best_f1_score:
                            best_f1_score = f1
                            is_best_step = True

                    best_flag = torch.tensor([1 if is_best_step else 0], device=rank, dtype=torch.int)
                    if is_distributed:
                        dist.broadcast(best_flag, src=0)
                    if best_flag.item() == 1:
                        save_fsdp_model(model, rank, config, epoch, is_best=True)
                            
                    model.train()

        metrics = evaluate(model, valid_loader, rank)
        is_best_epoch = False
        
        if rank == 0:
            val_loss = metrics["val_loss"]
            f1 = metrics["f1"]
            print(f"\n[info] Epoch {epoch} finished. Val Loss: {val_loss:.4f} | F1: {f1:.4f}")
            
            wandb.log({
                "val/loss": val_loss,
                "val/accuracy": metrics["accuracy"],
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "val/f1": f1,
                "epoch": epoch
            })
            
            if f1 > best_f1_score:
                best_f1_score = f1
                is_best_epoch = True

        best_epoch_flag = torch.tensor([1 if is_best_epoch else 0], device=rank, dtype=torch.int)
        if is_distributed:
            dist.broadcast(best_epoch_flag, src=0)
        if best_epoch_flag.item() == 1:
            save_fsdp_model(model, rank, config, epoch, is_best=True)

        save_fsdp_model(model, rank, config, epoch, is_best=False)