import csv
import json
import math
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from segment_dataset import seg_code_budget
from utils import setup, cleanup, load_model, get_raw_model, set_global_seed
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
)


class ELFFileTestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, config):
        self.tok = tokenizer
        self.budget = seg_code_budget(config)
        self.max_segments = int(config.max_segments_per_file)
        self.eos_id = self.tok.eos_token_id

        self.splits_base = config.data.splits_base
        self.corpus_base = config.data.corpus_base
        self.experiment_name = config.data.experiment_name
        self.platform = config.data.platform

        self.samples = self._index()

        imbalanced_ratio = getattr(config.data, "test_imbalanced_ratio", None)
        if imbalanced_ratio is not None and imbalanced_ratio > 0:
            self.samples = self._make_imbalanced(self.samples, int(imbalanced_ratio))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                n_ben = sum(1 for s in self.samples if s["label"] == 0)
                n_mal = sum(1 for s in self.samples if s["label"] == 1)
                print(f"[info] [dataset-elf] imbalanced subsampling {imbalanced_ratio}:1 -> benign={n_ben}, malware={n_mal}")

    def _index(self) -> list[dict]:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        csv_path = Path(self.splits_base) / self.experiment_name / "test.csv"
        samples = []
        if not csv_path.exists():
            if local_rank == 0:
                print(f"[warn] [dataset-elf] split csv not found: {csv_path}")
            return samples
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                label = int(row[0])
                sha256 = row[1]
                label_dir = "malware" if label == 1 else "benign"
                json_path = Path(self.corpus_base) / self.platform / label_dir / f"{sha256}.json"
                if json_path.exists():
                    samples.append({"path": str(json_path), "label": label})
        samples.sort(key=lambda x: x["path"])
        if local_rank == 0:
            print(f"[info] [dataset-elf] loaded {len(samples)} test samples from {csv_path}")
        return samples

    @staticmethod
    def _make_imbalanced(samples: list[dict], benign_ratio: int = 9) -> list[dict]:
        benign = [s for s in samples if s["label"] == 0]
        malware = [s for s in samples if s["label"] == 1]
        n_malware = int(len(benign) / benign_ratio)
        rng = random.Random(42)
        rng.shuffle(malware)
        imbalanced = benign + malware[:n_malware]
        rng.shuffle(imbalanced)
        return imbalanced

    def _load_code(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("decompiled_code") or ""
        except Exception:
            return ""

    def _segment(self, code: str) -> list[list[int]]:
        if not code:
            return [[self.eos_id]]
        ids = self.tok(
            code,
            add_special_tokens=False,
            truncation=True,
            max_length=self.budget * self.max_segments,
        )["input_ids"]
        if not ids:
            return [[self.eos_id]]
        n = min(self.max_segments, math.ceil(len(ids) / self.budget))
        return [ids[i * self.budget:(i + 1) * self.budget] for i in range(n)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        segs = self._segment(self._load_code(sample["path"]))
        max_len = max(len(s) for s in segs)

        code_ids = []
        code_masks = []
        for s in segs:
            ids = torch.tensor(s, dtype=torch.long)
            pad_n = max_len - ids.size(0)
            code_ids.append(torch.cat([ids.new_zeros(pad_n), ids]))
            code_masks.append(
                torch.cat([
                    torch.zeros(pad_n, dtype=torch.long),
                    torch.ones(ids.size(0), dtype=torch.long),
                ])
            )

        return {
            "code_ids": torch.stack(code_ids),
            "code_mask": torch.stack(code_masks),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def elf_test_collate_fn(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(f"file-level testing requires batch_size=1, got {len(batch)}")
    x = batch[0]
    return {
        "code_ids": x["code_ids"],
        "code_mask": x["code_mask"],
        "labels": x["label"].unsqueeze(0),
        "idx": x["idx"].unsqueeze(0),
    }


def build_elf_test_loader(config, tokenizer) -> torch.utils.data.DataLoader:
    test_ds = ELFFileTestDataset(tokenizer=tokenizer, config=config)

    _is_dist = dist.is_available() and dist.is_initialized()

    test_sampler = None
    if _is_dist:
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        sampler=test_sampler,
        collate_fn=elf_test_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"[info] [dataset-elf] test size: {len(test_ds)}")

    return test_loader


def eval_worker(rank, config):
    set_global_seed(int(getattr(config, "seed", 42)))
    setup(rank, config.world_size)

    imbalanced_ratio = getattr(config.data, "test_imbalanced_ratio", None)
    if imbalanced_ratio is not None and imbalanced_ratio > 0:
        ratio_tag = f"imbalanced_1_{int(imbalanced_ratio)}"
    else:
        ratio_tag = "balanced_1_1"

    model_short = config.model_id.split("/")[-1]
    run_name = f"{model_short}-encdec-test-elf-{ratio_tag}"

    if rank == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "LLM-Malware-Detection"),
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True),
            tags=["test", "elf", "encdec", ratio_tag, model_short],
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_loader = build_elf_test_loader(config, tokenizer)

        model = load_model(config=config, tokenizer=tokenizer, rank=rank)
        checkpoint_path = os.path.join(config.output_dir, config.test_checkpoint_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        if rank == 0:
            print(f"[{rank}] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["trainable_state_dict"]

        raw = get_raw_model(model)
        own_keys = set(raw.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in own_keys}
        skipped = [k for k in state_dict if k not in own_keys]
        raw.load_state_dict(filtered, strict=False)

        if rank == 0:
            print(f"[{rank}] Loaded {len(filtered)} trainable tensors | skipped: {len(skipped)}")
            if skipped:
                print(f"[{rank}] Skipped keys: {skipped[:5]}")

        model.eval()

        all_preds_local = []
        all_labels_local = []
        all_probs_local = []
        all_logits_local = []
        all_idx_local = []

        pbar = tqdm(test_loader, desc=f"Testing [Rank {rank}]") if rank == 0 else test_loader

        with torch.no_grad():
            for batch in pbar:
                code_ids = batch["code_ids"].to(rank)
                code_mask = batch["code_mask"].to(rank)
                labels = batch["labels"].to(rank)
                indices = batch["idx"].to(rank)

                out = model(code_ids=code_ids, code_mask=code_mask)
                logits = out["logits"]
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_logits_local.append(logits.float())
                all_probs_local.append(probs.float())
                all_preds_local.append(preds.float())
                all_labels_local.append(labels.float())
                all_idx_local.append(indices.long())

        logits_t = torch.cat(all_logits_local).float().cpu()
        probs_t = torch.cat(all_probs_local).float().cpu()
        preds_t = torch.cat(all_preds_local).float().cpu()
        labels_t = torch.cat(all_labels_local).float().cpu()
        idx_t = torch.cat(all_idx_local).long().cpu()

        local_data = {
            "logits": logits_t,
            "probs": probs_t,
            "preds": preds_t,
            "labels": labels_t,
            "idx": idx_t,
        }

        gathered_data = [None for _ in range(config.world_size)]
        dist.all_gather_object(gathered_data, local_data)

        if rank == 0:
            all_logits_raw = torch.cat([d["logits"] for d in gathered_data]).numpy()
            all_probs_raw = torch.cat([d["probs"] for d in gathered_data]).numpy()
            all_preds_raw = torch.cat([d["preds"] for d in gathered_data]).numpy()
            all_labels_raw = torch.cat([d["labels"] for d in gathered_data]).numpy()
            all_idx_raw = torch.cat([d["idx"] for d in gathered_data]).numpy()

            unique_indices = {}
            for i, idx in enumerate(all_idx_raw):
                if idx not in unique_indices:
                    unique_indices[idx] = i

            sorted_keys = sorted(unique_indices.keys())
            dedup_indices = [unique_indices[k] for k in sorted_keys]

            all_logits = all_logits_raw[dedup_indices]
            all_probs = all_probs_raw[dedup_indices]
            all_preds = all_preds_raw[dedup_indices]
            all_labels = all_labels_raw[dedup_indices]

            n_benign = int((all_labels == 0).sum())
            n_malware = int((all_labels == 1).sum())
            print(f"Evaluated exactly {len(all_probs)} unique samples (Test set size expected: {len(test_loader.dataset)}).")
            print(f"Label distribution: benign={n_benign}, malware={n_malware}")
            print(f"Prediction distribution: benign={int((all_preds == 0).sum())}, malware={int((all_preds == 1).sum())}")
            print(f"Prob range: min={all_probs.min():.4f}, max={all_probs.max():.4f}, mean={all_probs.mean():.4f}")

            try:
                acc = accuracy_score(all_labels, all_preds)
                prec = precision_score(all_labels, all_preds, zero_division=0)
                rec = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)

                fpr, tpr, _ = roc_curve(all_labels, all_probs)
                roc_auc_val = auc(fpr, tpr)

                prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
                pr_auc_val = auc(rec_curve, prec_curve)

                tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

                if np.any(tpr >= 0.95):
                    fpr_at_95rec = float(fpr[np.argmax(tpr >= 0.95)])
                else:
                    fpr_at_95rec = float("nan")

                valid = np.where(fpr <= 0.05)[0]
                rec_at_5fpr = float(tpr[valid[-1]]) if len(valid) > 0 else float("nan")

            except Exception as e:
                print(f"Error calculating metrics: {e}")
                acc, prec, rec, f1 = 0, 0, 0, 0
                roc_auc_val, pr_auc_val = 0, 0
                fpr_at_95rec, rec_at_5fpr = 0, 0
                tp, tn, fp, fn = 0, 0, 0, 0
                fpr, tpr, prec_curve, rec_curve = [], [], [], []

            metrics = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "roc_auc": float(roc_auc_val),
                "pr_auc": float(pr_auc_val),
                "fpr_at_95rec": float(fpr_at_95rec),
                "rec_at_5fpr": float(rec_at_5fpr),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "n_benign": n_benign,
                "n_malware": n_malware,
                "n_total": len(all_probs),
                "ratio_tag": ratio_tag,
            }

            print("\n--- Test Metrics ---")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
            print(classification_report(all_labels, all_preds, target_names=["benign", "malware"], zero_division=0))

            wandb.log({f"test/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

            wandb.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels.astype(int).tolist(),
                    preds=all_preds.astype(int).tolist(),
                    class_names=["benign", "malware"],
                )
            })

            roc_table = wandb.Table(
                data=[[float(x), float(y)] for x, y in zip(fpr, tpr)],
                columns=["FPR", "TPR"],
            )
            wandb.log({
                "test/roc_curve": wandb.plot.line(
                    roc_table, "FPR", "TPR",
                    title=f"ROC Curve (AUC={roc_auc_val:.4f})",
                )
            })

            pr_table = wandb.Table(
                data=[[float(r), float(p)] for r, p in zip(rec_curve, prec_curve)],
                columns=["Recall", "Precision"],
            )
            wandb.log({
                "test/pr_curve": wandb.plot.line(
                    pr_table, "Recall", "Precision",
                    title=f"PR Curve (AUC={pr_auc_val:.4f})",
                )
            })

            dataset = test_loader.dataset
            results_table = wandb.Table(columns=["filename", "logit", "label", "pred", "prob"])
            for i, idx in enumerate(sorted_keys):
                sample = dataset.samples[int(idx)]
                filename = Path(sample["path"]).name
                results_table.add_data(
                    filename,
                    float(all_logits[i]),
                    int(all_labels[i]),
                    int(all_preds[i]),
                    float(all_probs[i]),
                )
            wandb.log({"test/predictions": results_table})

            os.makedirs(config.output_dir, exist_ok=True)

            out_file = os.path.join(config.output_dir, f"test_metrics_elf_{ratio_tag}.json")
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"\nMetrics saved to {out_file}")

            csv_file = os.path.join(config.output_dir, f"test_results_elf_{ratio_tag}.csv")
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "logit", "label", "pred", "prob"])
                for i, idx in enumerate(sorted_keys):
                    sample = dataset.samples[int(idx)]
                    filename = Path(sample["path"]).name
                    writer.writerow([
                        filename,
                        float(all_logits[i]),
                        int(all_labels[i]),
                        int(all_preds[i]),
                        float(all_probs[i]),
                    ])
            print(f"Results saved to {csv_file}")

    finally:
        cleanup()
        if rank == 0:
            wandb.finish()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    model_cfg = cfg.model
    print(f"Starting test script with {model_cfg.world_size} GPUs...")
    mp.spawn(
        eval_worker,
        args=(model_cfg,),
        nprocs=model_cfg.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
