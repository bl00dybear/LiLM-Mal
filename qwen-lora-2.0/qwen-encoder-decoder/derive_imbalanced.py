import argparse
import csv
import json
import os
import random
import numpy as np

import wandb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
)


def read_results(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "filename": r["filename"],
                "logit": float(r["logit"]),
                "label": int(r["label"]),
                "prob": float(r["prob"]),
            })
    return rows


def make_imbalanced(rows, benign_ratio):
    benign = [r for r in rows if r["label"] == 0]
    malware = [r for r in rows if r["label"] == 1]
    malware.sort(key=lambda r: r["filename"])
    n_malware = int(len(benign) / benign_ratio)
    rng = random.Random(42)
    rng.shuffle(malware)
    return benign + malware[:n_malware]


def compute_metrics(subset):
    labels = np.array([r["label"] for r in subset])
    probs = np.array([r["prob"] for r in subset])
    preds = (probs > 0.5).astype(float)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc_val = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(labels, probs)
    pr_auc_val = auc(rec_curve, prec_curve)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    fpr_at_95rec = float(fpr[np.argmax(tpr >= 0.95)]) if np.any(tpr >= 0.95) else float("nan")
    valid = np.where(fpr <= 0.05)[0]
    rec_at_5fpr = float(tpr[valid[-1]]) if len(valid) > 0 else float("nan")

    metrics = {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "roc_auc": float(roc_auc_val), "pr_auc": float(pr_auc_val),
        "fpr_at_95rec": float(fpr_at_95rec), "rec_at_5fpr": float(rec_at_5fpr),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_benign": int((labels == 0).sum()), "n_malware": int((labels == 1).sum()),
        "n_total": int(len(labels)),
    }
    return metrics, labels, preds, probs, (fpr, tpr), (prec_curve, rec_curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--ratio", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--arch", default="encdec")
    args = ap.parse_args()

    ratio_tag = f"imbalanced_1_{args.ratio}"
    rows = read_results(args.results_csv)
    subset = make_imbalanced(rows, args.ratio)
    metrics, labels, preds, probs, (fpr, tpr), (prec_curve, rec_curve) = compute_metrics(subset)
    metrics["ratio_tag"] = ratio_tag

    print(f"Derived {ratio_tag} from {args.results_csv}: {len(subset)} samples "
          f"(benign={metrics['n_benign']}, malware={metrics['n_malware']})")
    print("\n--- Test Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(classification_report(labels, preds, target_names=["benign", "malware"], zero_division=0))

    model_short = args.model_id.split("/")[-1]
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "LLM-Malware-Detection"),
        name=f"{model_short}-{args.arch}-test-elf-{ratio_tag}",
        tags=["test", "elf", args.arch, ratio_tag, model_short, "derived"],
    )
    wandb.log({f"test/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=labels.astype(int).tolist(),
            preds=preds.astype(int).tolist(), class_names=["benign", "malware"],
        )
    })
    roc_table = wandb.Table(data=[[float(x), float(y)] for x, y in zip(fpr, tpr)], columns=["FPR", "TPR"])
    wandb.log({"test/roc_curve": wandb.plot.line(roc_table, "FPR", "TPR", title=f"ROC (AUC={metrics['roc_auc']:.4f})")})
    pr_table = wandb.Table(data=[[float(r), float(p)] for r, p in zip(rec_curve, prec_curve)], columns=["Recall", "Precision"])
    wandb.log({"test/pr_curve": wandb.plot.line(pr_table, "Recall", "Precision", title=f"PR (AUC={metrics['pr_auc']:.4f})")})

    os.makedirs(args.output_dir, exist_ok=True)
    json_file = os.path.join(args.output_dir, f"test_metrics_elf_{ratio_tag}.json")
    with open(json_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {json_file}")

    csv_file = os.path.join(args.output_dir, f"test_results_elf_{ratio_tag}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "logit", "label", "pred", "prob"])
        for r in subset:
            writer.writerow([r["filename"], r["logit"], r["label"], int(r["prob"] > 0.5), r["prob"]])
    print(f"Results saved to {csv_file}")

    wandb.finish()


if __name__ == "__main__":
    main()
