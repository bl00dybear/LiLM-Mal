from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

from config import ThresholdConfig


def load_predictions(csv_path: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    return df['label'], df['prob']


def compute_pr_curve(y_true: pd.Series, y_prob: pd.Series) -> tuple:
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    return precision_curve, recall_curve, pr_auc


def find_best_threshold(y_true: pd.Series, y_prob: pd.Series, pr_auc: float) -> dict:
    best_f1 = 0.0
    best_metrics = {}

    for thresh in np.arange(0.0, 1.01, 0.001):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'Threshold': round(thresh, 2),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall':    recall_score(y_true, y_pred, zero_division=0),
                'F1-Score':  f1,
                'PR-AUC':    pr_auc,
            }

    return best_metrics


def model_label(csv_path: str) -> str:
    parts = csv_path.replace('outputs/', '').split('/')
    checkpoint = parts[0].replace('checkpoints-', '')
    split = parts[1].replace('test_results_', '').replace('.csv', '')
    return f"{checkpoint} / {split}"


def process_single(csv_path: str) -> dict:
    y_true, y_prob = load_predictions(csv_path)
    precision_curve, recall_curve, pr_auc = compute_pr_curve(y_true, y_prob)
    metrics = find_best_threshold(y_true, y_prob, pr_auc)
    metrics['Model'] = model_label(csv_path)
    metrics['_path'] = csv_path
    metrics['_precision_curve'] = precision_curve.tolist()
    metrics['_recall_curve'] = recall_curve.tolist()
    return metrics


def run_parallel(csv_paths: list[str], max_workers: int | None = None) -> list[dict]:
    results = [None] * len(csv_paths)
    index_map = {path: i for i, path in enumerate(csv_paths)}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, path): path for path in csv_paths}

        for future in as_completed(futures):
            path = futures[future]
            result = future.result()
            results[index_map[path]] = result

    return results


def plot_pr_grid(results: list[dict], output_path: str) -> None:
    n_cols = 2
    n_rows = (len(results) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3.5))
    axes = axes.flatten()

    for i, result in enumerate(results):
        precision_curve = np.array(result['_precision_curve'])
        recall_curve    = np.array(result['_recall_curve'])
        pr_auc          = result['PR-AUC']

        ax = axes[i]
        ax.plot(recall_curve, precision_curve, color='steelblue', lw=1.8)
        ax.set_title(result['Model'], fontsize=9, fontweight='bold')
        ax.set_xlabel('Recall', fontsize=8)
        ax.set_ylabel('Precision', fontsize=8)
        ax.legend([f'PR-AUC = {pr_auc:.4f}'], fontsize=8, loc='lower left')
        ax.grid(True, linewidth=0.4)
        ax.tick_params(labelsize=7)

    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def build_metrics_df(results: list[dict]) -> pd.DataFrame:
    rows = [
        {k: v for k, v in r.items() if not k.startswith('_')}
        for r in results
    ]
    df = pd.DataFrame(rows)
    return df[['Model', 'Threshold', 'Precision', 'Recall', 'F1-Score', 'PR-AUC']]


def main() -> None:
    config = ThresholdConfig()

    results = run_parallel(config.models_csvs)

    plot_pr_grid(results, output_path='outputs/pr_curves.pdf')

    metrics_df = build_metrics_df(results)
    metrics_df.to_csv('outputs/best_metrics.csv', index=False)

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()