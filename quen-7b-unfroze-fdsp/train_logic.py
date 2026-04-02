import torch
import torch.distributed as dist
from torch.optim import Adam
from transformers import get_cosine_schedule_with_warmup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import math
import os

from tqdm.auto import tqdm
from plotting import plot_step_metrics, plot_epoch_metrics, plot_confusion_matrix, ensure_plot_dirs


def evaluate(model, val_loader, device_id, is_distributed) -> dict | None:
    num_classes = 2
    model.eval()
    try:
        total_loss  = torch.tensor(0.0, device=device_id)
        num_batches = torch.tensor(0,   device=device_id, dtype=torch.long)
        correct     = torch.tensor(0,   device=device_id, dtype=torch.long)
        total       = torch.tensor(0,   device=device_id, dtype=torch.long)
        tp = torch.zeros(num_classes, device=device_id, dtype=torch.long)
        fp = torch.zeros(num_classes, device=device_id, dtype=torch.long)
        fn = torch.zeros(num_classes, device=device_id, dtype=torch.long)

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device_id)
                attention_mask = batch["attention_mask"].to(device_id)
                labels         = batch["labels"].to(device_id)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss    = outputs["loss"]
                preds   = outputs["logits"].argmax(dim=-1)
                total_loss  += loss.detach().float()
                num_batches += 1
                correct     += (preds == labels).sum()
                total       += labels.size(0)
                for c in range(num_classes):
                    tp[c] += ((preds == c) & (labels == c)).sum()
                    fp[c] += ((preds == c) & (labels != c)).sum()
                    fn[c] += ((preds != c) & (labels == c)).sum()

        if is_distributed:
            for t in [total_loss, num_batches, correct, total, tp, fp, fn]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if dist.get_rank() != 0:
                return None

        val_loss = (total_loss / num_batches.clamp(min=1)).item()
        val_acc  = (correct.float() / total.clamp(min=1)).item()

        f1_per_class = []
        for c in range(num_classes):
            prec = tp[c].float() / (tp[c] + fp[c]).float().clamp(min=1)
            rec  = tp[c].float() / (tp[c] + fn[c]).float().clamp(min=1)
            f1_c = 2.0 * prec * rec / (prec + rec).clamp(min=1e-8)
            f1_per_class.append(f1_c.item())
        val_f1 = sum(f1_per_class) / len(f1_per_class)

        return {
            "val_loss": val_loss, "val_accuracy": val_acc, "val_f1": val_f1,
            "tp": tp[1].item(), "tn": tp[0].item(),
            "fp": fp[1].item(), "fn": fn[1].item(),
        }
    finally:
        model.train()


def train_model(model, train_loader, val_loader, config, is_distributed, device_id):
    model.train()

    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    if device_id == 0:
        print(
            f"[Optimizer] lr={config.learning_rate} | "
            f"param_groups={len(optimizer.param_groups)} | "
            f"params_with_grad={sum(1 for p in model.parameters() if p.requires_grad)}"
        )

    epochs           = getattr(config, "epochs", 3)
    grad_accum_steps = getattr(config, "grad_accum_steps", 1)
    world_size       = int(os.environ.get("WORLD_SIZE", 1))

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = max(1, int(0.1 * total_steps))

    if device_id == 0:
        print(
            f"[Scheduler] world_size={world_size} | loader_len={len(train_loader)} | "
            f"grad_accum={grad_accum_steps} | steps_per_epoch={steps_per_epoch} | "
            f"total_steps={total_steps} | warmup_steps={warmup_steps} | epochs={epochs}"
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    plot_dir = getattr(config, "plot_dir", "outputs/plots")
    if device_id == 0:
        ensure_plot_dirs(plot_dir)

    step_history = {
        "steps": [], "train_loss": [], "lr": [], "grad_norm": [],
        "warmup_end_step": warmup_steps,
    }
    epoch_history = {
        "epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": [],
    }
    global_step   = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        if is_distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch}", disable=is_distributed and device_id != 0
        )

        for step, batch in progress_bar:
            input_ids      = batch["input_ids"].to(device_id)
            attention_mask = batch["attention_mask"].to(device_id)
            labels         = batch["labels"].to(device_id)

            is_sync_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))

            from contextlib import nullcontext
            sync_context = nullcontext()
            if is_distributed and not is_sync_step:
                if hasattr(model, "no_sync"):
                    sync_context = model.no_sync()

            with sync_context:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss    = outputs["loss"]
                loss    = loss / grad_accum_steps
                loss.backward()

            step_loss   = loss.item() * grad_accum_steps
            total_loss += step_loss

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if is_distributed:
                    grad_norm = model.clip_grad_norm_(1.0)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if device_id == 0:
                    step_history["steps"].append(global_step)
                    step_history["train_loss"].append(step_loss)
                    step_history["lr"].append(scheduler.get_last_lr()[0])
                    step_history["grad_norm"].append(
                        grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                    )

            if step % 5 == 0:
                progress_bar.set_postfix({
                    "loss": f"{step_loss:.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.2e}"
                })

            if step > 0 and step % getattr(config, "save_every_n_steps", 500) == 0:
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    cpu_state = model.state_dict()

                if not (is_distributed and device_id != 0):
                    if hasattr(config, "output_dir"):
                        os.makedirs(config.output_dir, exist_ok=True)
                        save_path = os.path.join(config.output_dir, "qwen_malware_latest_step.pt")
                        torch.save(cpu_state, save_path)
                        progress_bar.write(f">> Salvare Periodica: pasul {step+1} → {save_path}")
                    plot_step_metrics(step_history, plot_dir)
                    progress_bar.write(f">> Ploturi step actualizate (step {step+1})")

        avg_loss = total_loss / len(train_loader)
        if not (is_distributed and device_id != 0):
            progress_bar.write(f"Epoch {epoch} complete | Average Loss: {avg_loss:.4f}")

        metrics = evaluate(model, val_loader, device_id, is_distributed)

        if metrics is not None:
            progress_bar.write(
                f"[Val] Epoch {epoch} | Loss: {metrics['val_loss']:.4f} | "
                f"Acc: {metrics['val_accuracy']:.4f} | F1: {metrics['val_f1']:.4f}"
            )
            epoch_history["epochs"].append(epoch)
            epoch_history["train_loss"].append(avg_loss)
            epoch_history["val_loss"].append(metrics["val_loss"])
            epoch_history["val_accuracy"].append(metrics["val_accuracy"])
            epoch_history["val_f1"].append(metrics["val_f1"])

            plot_epoch_metrics(epoch_history, plot_dir)
            plot_confusion_matrix(
                tp=metrics["tp"], tn=metrics["tn"],
                fp=metrics["fp"], fn=metrics["fn"],
                epoch=epoch, plot_dir=plot_dir,
            )
            progress_bar.write(f">> Ploturi epocă actualizate (epoch {epoch})")

        is_best = torch.zeros(1, device=device_id)
        if metrics is not None and metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            is_best[0]    = 1.0
        if is_distributed:
            dist.broadcast(is_best, src=0)

        if is_best.item() > 0:
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                best_cpu_state = model.state_dict()
            if not (is_distributed and device_id != 0):
                if hasattr(config, "output_dir"):
                    os.makedirs(config.output_dir, exist_ok=True)
                    best_name = getattr(config, "best_checkpoint_name", "qwen_malware_best.pt")
                    best_path = os.path.join(config.output_dir, best_name)
                    torch.save(best_cpu_state, best_path)
                    progress_bar.write(
                        f">> Best checkpoint salvat la epoca {epoch} | Val Loss: {metrics['val_loss']:.4f}"
                    )

        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()

        if not (is_distributed and device_id != 0):
            if hasattr(config, "output_dir"):
                os.makedirs(config.output_dir, exist_ok=True)
                save_path = os.path.join(config.output_dir, f"qwen_malware_epoch_{epoch}.pt")
                torch.save(cpu_state, save_path)
                progress_bar.write(f">> Checkpoint epoca {epoch} salvat in {save_path}\n")

    if device_id == 0:
        plot_step_metrics(step_history, plot_dir)

    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        final_cpu_state = model.state_dict()

    if not (is_distributed and device_id != 0):
        if hasattr(config, "output_dir"):
            final_save_path = os.path.join(config.output_dir, "qwen_malware_final.pt")
            torch.save(final_cpu_state, final_save_path)
            print(f"[info] training completed. model saved in {final_save_path}")
