from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


@dataclass
class ModelPaths:
    label_prefix: str
    val_csv: str
    test_csvs: list[str] = field(default_factory=list)


@dataclass
class ThresholdConfig:
    models: list[ModelPaths] = field(default_factory=lambda: [
        ModelPaths(
            label_prefix="q1.5b",
            val_csv="outputs/checkpoints-q1.5b/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q1.5b/test_results_elf.csv",
                "outputs/checkpoints-q1.5b/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q1.5b-lora-classic",
            val_csv="outputs/checkpoints-q1.5b-lora-classic/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q1.5b-lora-classic/test_results_elf.csv",
                "outputs/checkpoints-q1.5b-lora-classic/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q1.5b-lora-attention",
            val_csv="outputs/checkpoints-q1.5b-lora-attention/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q1.5b-lora-attention/test_results_elf.csv",
                "outputs/checkpoints-q1.5b-lora-attention/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q1.5b-lora-full-19M",
            val_csv="outputs/checkpoints-q1.5b-lora-full-19M/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q1.5b-lora-full-19M/test_results_elf.csv",
                "outputs/checkpoints-q1.5b-lora-full-19M/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q1.5b-lora-full-38M",
            val_csv="outputs/checkpoints-q1.5b-lora-full-38M/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q1.5b-lora-full-38M/test_results_elf.csv",
                "outputs/checkpoints-q1.5b-lora-full-38M/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q1.5b-lora-full-76M",
            val_csv="outputs/checkpoints-q1.5b-lora-full-76M/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q1.5b-lora-full-76M/test_results_elf.csv",
                "outputs/checkpoints-q1.5b-lora-full-76M/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q3b",
            val_csv="outputs/checkpoints-q3b/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q3b/test_results_elf.csv",
                "outputs/checkpoints-q3b/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q3b-lora-classic",
            val_csv="outputs/checkpoints-q3b-lora-classic/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q3b-lora-classic/test_results_elf.csv",
                "outputs/checkpoints-q3b-lora-classic/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q3b-lora-attention",
            val_csv="outputs/checkpoints-q3b-lora-attention/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q3b-lora-attention/test_results_elf.csv",
                "outputs/checkpoints-q3b-lora-attention/test_results_pe.csv",
            ],
        ),
        ModelPaths(
            label_prefix="q3b-lora-full",
            val_csv="outputs/checkpoints-q3b-lora-full/val_results_elf_1_9.csv",
            test_csvs=[
                "outputs/checkpoints-q3b-lora-full/test_results_elf_1_9.csv",
                "outputs/checkpoints-q3b-lora-full/test_results_pe.csv",
            ],
        ),
    ])


def load_csv(path: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    return df["label"], df["prob"]


def compute_pr_curve(y_true: pd.Series, y_prob: pd.Series) -> tuple:
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    return precision_curve, recall_curve, pr_auc


def find_best_threshold(y_true: pd.Series, y_prob: pd.Series) -> tuple[float, float]:
    best_f1 = 0.0
    best_thresh = 0.5

    for thresh in np.arange(0.0, 1.01, 0.001):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = round(float(thresh), 3)

    return best_thresh, best_f1


def evaluate_on_test(
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    _, _, pr_auc = compute_pr_curve(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    
    target_recall = 0.95
    if np.any(tpr >= target_recall):
        idx_fixed_recall = np.argmax(tpr >= target_recall)
        fpr_at_fixed_recall = fpr[idx_fixed_recall]
    else:
        fpr_at_fixed_recall = np.nan
        
    target_fpr = 0.05
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) > 0:
        recall_at_low_fpr = tpr[valid_indices[-1]]
    else:
        recall_at_low_fpr = np.nan

    return {
        "Threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
        "PR-AUC":    pr_auc,
        "ROC-AUC":   roc_auc_val,
        "FPR@95%Rec": fpr_at_fixed_recall,
        "Rec@5%FPR": recall_at_low_fpr,
        "_precision_curve": precision_curve.tolist(),
        "_recall_curve":    recall_curve.tolist(),
    }


def test_label(label_prefix: str, test_csv: str) -> str:
    name = Path(test_csv).stem
    split = name.replace("test_results_", "").replace("_1_9", "")
    return f"{label_prefix} / {split}"


def process_single(model: ModelPaths) -> list[dict]:
    y_val_true, y_val_prob = load_csv(model.val_csv)
    best_thresh, best_val_f1 = find_best_threshold(y_val_true, y_val_prob)

    results = []

    for test_csv in model.test_csvs:
        y_test_true, y_test_prob = load_csv(test_csv)
        metrics = evaluate_on_test(y_test_true, y_test_prob, best_thresh)

        metrics["Model"]   = test_label(model.label_prefix, test_csv)
        metrics["Val-F1"]  = round(best_val_f1, 4)
        metrics["_path"]   = test_csv
        results.append(metrics)

    return results


def run_parallel(models: list[ModelPaths], max_workers: int | None = None) -> list[dict]:
    flat_results: list[dict | None] = [None] * len(models)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, m): i for i, m in enumerate(models)}
        for future in as_completed(futures):
            idx = futures[future]
            flat_results[idx] = future.result()

    return [r for sublist in flat_results for r in sublist]


def plot_pr_grid(results: list[dict], output_path: str) -> None:
    n_cols = 2
    n_rows = (len(results) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3.5))
    axes = axes.flatten()

    for i, result in enumerate(results):
        precision_curve = np.array(result["_precision_curve"])
        recall_curve    = np.array(result["_recall_curve"])
        pr_auc          = result["PR-AUC"]

        ax = axes[i]
        ax.plot(recall_curve, precision_curve, color="steelblue", lw=1.8)
        ax.set_title(result["Model"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.legend([f"PR-AUC = {pr_auc:.4f}"], fontsize=8, loc="lower left")
        ax.grid(True, linewidth=0.4)
        ax.tick_params(labelsize=7)

    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def build_metrics_df(results: list[dict]) -> pd.DataFrame:
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    df = pd.DataFrame(rows)
    col_order = [
        "Model", "Threshold", "Val-F1", 
        "TP", "TN", "FP", "FN", 
        "Precision", "Recall", "F1-Score", "PR-AUC", "ROC-AUC",
        "FPR@95%Rec", "Rec@5%FPR"
    ]
    return df[[c for c in col_order if c in df.columns]]


def main() -> None:
    config = ThresholdConfig()

    results = run_parallel(config.models)

    plot_pr_grid(results, output_path="outputs/pr_curves.pdf")

    metrics_df = build_metrics_df(results)
    metrics_df.to_csv("outputs/best_metrics.csv", index=False)

    print("\n" + metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()