import os
import json
import csv
import random
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier

SPLITS_BASE = Path("/run/media/sebi/nvme-1tb/LiLM-Mal-Dataset/data/02_metadata/experiments")
CORPUS_BASE = Path("/run/media/sebi/nvme-1tb/LiLM-Mal-Dataset/data/03_corpus_v2")
SEED = 42
N_JOBS = 40


def load_from_csv(experiment_name: str, partition: str, platform: str) -> list[dict]:
    csv_path = SPLITS_BASE / experiment_name / f"{partition}.csv"
    samples = []
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return samples
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = int(row[0])
            sha256 = row[1]
            label_dir = "malware" if label == 1 else "benign"
            json_path = CORPUS_BASE / platform / label_dir / f"{sha256}.json"
            if json_path.exists():
                samples.append({"path": str(json_path), "label": label})
    samples.sort(key=lambda x: x["path"])
    return samples


def _process_single_raw(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("decompiled_code") or ""
    except Exception:
        return ""


def extract_texts_parallel(samples: list[dict], desc: str) -> tuple[list[str], np.ndarray]:
    paths = [s["path"] for s in samples]
    labels = np.array([s["label"] for s in samples])
    texts = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        batch_size=64,
        verbose=0,
    )(
        delayed(_process_single_raw)(p)
        for p in tqdm(paths, desc=desc, unit="file")
    )
    return texts, labels


def split_train_val(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    benign_idx = [i for i, s in enumerate(samples) if s["label"] == 0]
    malware_idx = [i for i, s in enumerate(samples) if s["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(benign_idx)
    rng.shuffle(malware_idx)
    n_val_benign = int(len(benign_idx) * 0.10)
    n_val_malware = int(len(malware_idx) * 0.10)
    val_indices = benign_idx[:n_val_benign] + malware_idx[:n_val_malware]
    train_indices = benign_idx[n_val_benign:] + malware_idx[n_val_malware:]
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    return train_samples, val_samples


def make_imbalanced(samples: list[dict], benign_ratio: int = 9) -> list[dict]:
    benign = [s for s in samples if s["label"] == 0]
    malware = [s for s in samples if s["label"] == 1]
    n_malware = int(len(benign) / benign_ratio)
    rng = random.Random(SEED)
    rng.shuffle(malware)
    imbalanced_samples = benign + malware[:n_malware]
    rng.shuffle(imbalanced_samples)
    return imbalanced_samples


def compute_advanced_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(rec_curve, prec_curve)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    target_recall = 0.95
    if np.any(tpr >= target_recall):
        idx_fixed = np.argmax(tpr >= target_recall)
        fpr_at_95rec = float(fpr[idx_fixed])
    else:
        fpr_at_95rec = float('nan')

    target_fpr = 0.05
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) > 0:
        rec_at_5fpr = float(tpr[valid[-1]])
    else:
        rec_at_5fpr = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "fpr_at_95rec": fpr_at_95rec,
        "rec_at_5fpr": rec_at_5fpr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    metrics = compute_advanced_metrics(y_true, y_prob, threshold)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Threshold:   {metrics['threshold']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"  FPR@95%Rec:  {metrics['fpr_at_95rec']:.4f}")
    print(f"  Rec@5%FPR:   {metrics['rec_at_5fpr']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print()
    print(classification_report(y_true, y_pred, target_names=["benign", "malware"], zero_division=0))
    print(confusion_matrix(y_true, y_pred))
    print()
    return metrics


def main():
    print(f"[config] using {N_JOBS} parallel workers on {os.cpu_count()} available CPUs")

    print("[1/5] loading dataset splits")
    full_train = load_from_csv("elf_v2_full", "train", "elf")
    train_samples, val_samples_1_1 = split_train_val(full_train)

    val_samples_1_9 = make_imbalanced(val_samples_1_1, benign_ratio=9)
    test_samples_1_1_elf = load_from_csv("elf_v2_full", "test", "elf")
    test_samples_1_9_elf = make_imbalanced(test_samples_1_1_elf, benign_ratio=9)

    print(f"  train: {len(train_samples)} | val (1:9): {len(val_samples_1_9)}")
    print(f"  test (1:1 ELF): {len(test_samples_1_1_elf)}")
    print(f"  test (1:9 ELF): {len(test_samples_1_9_elf)}")

    print("[2/5] extracting raw texts (parallel)")
    train_texts, y_train = extract_texts_parallel(train_samples, "  train texts")
    val_texts_1_9, y_val_1_9 = extract_texts_parallel(val_samples_1_9, "  val 1:9 texts")
    test_texts_1_1_elf, y_test_1_1_elf = extract_texts_parallel(test_samples_1_1_elf, "  test 1:1 ELF texts")
    test_texts_1_9_elf, y_test_1_9_elf = extract_texts_parallel(test_samples_1_9_elf, "  test 1:9 ELF texts")

    print("[3/5] building tfidf features")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-zA-Z0-9_]+",
        max_features=2000,
        sublinear_tf=False,
        use_idf=False,
        binary=True,
        min_df=10,
        max_df=0.9,
        ngram_range=(1, 1),
    )
    X_train_raw = vectorizer.fit_transform(train_texts)
    X_val_1_9_raw = vectorizer.transform(val_texts_1_9)
    X_test_1_1_elf_raw = vectorizer.transform(test_texts_1_1_elf)
    X_test_1_9_elf_raw = vectorizer.transform(test_texts_1_9_elf)

    selector = SelectKBest(score_func=chi2, k=50)
    X_train = selector.fit_transform(X_train_raw, y_train)
    X_val_1_9 = selector.transform(X_val_1_9_raw)
    X_test_1_1_elf = selector.transform(X_test_1_1_elf_raw)
    X_test_1_9_elf = selector.transform(X_test_1_9_elf_raw)

    print(f"  tfidf features reduced to: {X_train.shape[1]}")

    print("[4/5] training xgboost")
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=SEED,
        verbosity=1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val_1_9, y_val_1_9)],
        verbose=50,
    )

    print("\n[5/5] evaluating standard baselines (Threshold = 0.5)")

    test_prob_1_1_elf = clf.predict_proba(X_test_1_1_elf)[:, 1]
    test_pred_1_1_elf = (test_prob_1_1_elf >= 0.5).astype(int)
    m_1_1_elf = evaluate("Test (1:1 ELF Balanced) [T=0.5]", y_test_1_1_elf, test_pred_1_1_elf, test_prob_1_1_elf, threshold=0.5)

    test_prob_1_9_elf = clf.predict_proba(X_test_1_9_elf)[:, 1]
    test_pred_1_9_elf = (test_prob_1_9_elf >= 0.5).astype(int)
    m_1_9_elf_default = evaluate("Test (1:9 ELF Imbalanced) [T=0.5]", y_test_1_9_elf, test_pred_1_9_elf, test_prob_1_9_elf, threshold=0.5)

    print("\n--- Running Threshold Tuning on Validation (1:9 ELF) ---")
    val_prob_1_9 = clf.predict_proba(X_val_1_9)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.0, 1.001, 0.001):
        preds = (val_prob_1_9 >= thresh).astype(int)
        f1 = f1_score(y_val_1_9, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = round(float(thresh), 3)

    print(f"  Best threshold found on Validation 1:9: {best_thresh:.4f} (F1={best_f1:.4f})")

    print("\n[5/5] re-evaluating imbalanced configurations with optimized threshold")

    test_pred_1_9_elf_tuned = (test_prob_1_9_elf >= best_thresh).astype(int)
    m_1_9_elf_tuned = evaluate(f"Test (1:9 ELF Imbalanced) [Optimized T={best_thresh:.4f}]", y_test_1_9_elf, test_pred_1_9_elf_tuned, test_prob_1_9_elf, threshold=best_thresh)

    print("\n" + "="*80)
    print("  SUMMARY TABLE (matching paper format)")
    print("="*80)
    header = f"{'Split':<35} {'Thr':>6} {'F1':>8} {'PR-AUC':>8} {'FPR@95%Rec':>11} {'Rec@5%FPR':>10}"
    print(header)
    print("-" * len(header))
    for label, m in [
        ("ELF 1:9 (T=0.5)", m_1_9_elf_default),
        (f"ELF 1:9 (T={best_thresh})", m_1_9_elf_tuned),
    ]:
        print(f"  {label:<33} {m['threshold']:>6.3f} {m['f1']:>8.4f} {m['pr_auc']:>8.4f} {m['fpr_at_95rec']:>11.4f} {m['rec_at_5fpr']:>10.4f}")
    print()

    os.makedirs("outputs", exist_ok=True)
    all_results = {
        "elf_1_1_t05": m_1_1_elf,
        "elf_1_9_t05": m_1_9_elf_default,
        "best_threshold": best_thresh,
        "best_val_f1": best_f1,
        "elf_1_9_tuned": m_1_9_elf_tuned,
    }
    with open("outputs/baseline_xgboost_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("Metrics saved to outputs/baseline_xgboost_metrics.json")


if __name__ == "__main__":
    main()


# [5/5] evaluating standard baselines (Threshold = 0.5)

# ============================================================
#   Test (1:1 ELF Balanced) [T=0.5]
# ============================================================
#   Threshold:   0.5000
#   Accuracy:    0.9250
#   Precision:   0.9882
#   Recall:      0.8603
#   F1:          0.9198
#   ROC-AUC:     0.9825
#   PR-AUC:      0.9851
#   FPR@95%Rec:  0.2051
#   Rec@5%FPR:   0.8651
#   TP=7100  TN=8161  FP=85  FN=1153

#               precision    recall  f1-score   support

#       benign       0.88      0.99      0.93      8246
#      malware       0.99      0.86      0.92      8253

#     accuracy                           0.92     16499
#    macro avg       0.93      0.92      0.92     16499
# weighted avg       0.93      0.92      0.92     16499

# [[8161   85]
#  [1153 7100]]


# ============================================================
#   Test (1:9 ELF Imbalanced) [T=0.5]
# ============================================================
#   Threshold:   0.5000
#   Accuracy:    0.9772
#   Precision:   0.9031
#   Recall:      0.8646
#   F1:          0.8834
#   ROC-AUC:     0.9827
#   PR-AUC:      0.9403
#   FPR@95%Rec:  0.2051
#   Rec@5%FPR:   0.8712
#   TP=792  TN=8161  FP=85  FN=124

#               precision    recall  f1-score   support

#       benign       0.99      0.99      0.99      8246
#      malware       0.90      0.86      0.88       916

#     accuracy                           0.98      9162
#    macro avg       0.94      0.93      0.94      9162
# weighted avg       0.98      0.98      0.98      9162

# [[8161   85]
#  [ 124  792]]


# --- Running Threshold Tuning on Validation (1:9 ELF) ---
#   Best threshold found on Validation 1:9: 0.8960 (F1=0.9168)

# [5/5] re-evaluating imbalanced configurations with optimized threshold

# ============================================================
#   Test (1:9 ELF Imbalanced) [Optimized T=0.8960]
# ============================================================
#   Threshold:   0.8960
#   Accuracy:    0.9816
#   Precision:   0.9807
#   Recall:      0.8319
#   F1:          0.9002
#   ROC-AUC:     0.9827
#   PR-AUC:      0.9403
#   FPR@95%Rec:  0.2051
#   Rec@5%FPR:   0.8712
#   TP=762  TN=8231  FP=15  FN=154

#               precision    recall  f1-score   support

#       benign       0.98      1.00      0.99      8246
#      malware       0.98      0.83      0.90       916

#     accuracy                           0.98      9162
#    macro avg       0.98      0.92      0.95      9162
# weighted avg       0.98      0.98      0.98      9162

# [[8231   15]
#  [ 154  762]]


# ================================================================================
#   SUMMARY TABLE (matching paper format)
# ================================================================================
# Split                                  Thr       F1   PR-AUC  FPR@95%Rec  Rec@5%FPR
# -----------------------------------------------------------------------------------
#   ELF 1:9 (T=0.5)                    0.500   0.8834   0.9403      0.2051     0.8712
#   ELF 1:9 (T=0.896)                  0.896   0.9002   0.9403      0.2051     0.8712