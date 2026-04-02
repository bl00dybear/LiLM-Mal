import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def ensure_plot_dirs(plot_dir: str) -> None:
    os.makedirs(os.path.join(plot_dir, "step_plots"),  exist_ok=True)
    os.makedirs(os.path.join(plot_dir, "epoch_plots"), exist_ok=True)


def plot_step_metrics(history: dict, plot_dir: str) -> None:
    steps      = history.get("steps",      [])
    train_loss = history.get("train_loss", [])
    lr_vals    = history.get("lr",         [])
    grad_norm  = history.get("grad_norm",  [])
    warmup_end = history.get("warmup_end_step", None)
    step_dir   = os.path.join(plot_dir, "step_plots")

    if len(steps) >= 2 and len(train_loss) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, train_loss, color="steelblue", linewidth=1.0, label="Train Loss")
        ax.axhline(y=0.693, color="tomato", linestyle="--", linewidth=1.0, label="Random baseline (ln2)")
        ax.set_title("Train Loss per Step"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "loss_steps.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

    if len(steps) >= 2 and len(lr_vals) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, lr_vals, color="darkorange", linewidth=1.0, label="Learning Rate")
        if warmup_end is not None:
            ax.axvline(x=warmup_end, color="gray", linestyle="--", linewidth=1.0,
                       label=f"Warmup end (step {warmup_end})")
        ax.set_title("Learning Rate Schedule"); ax.set_xlabel("Step"); ax.set_ylabel("LR")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
        ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "lr_steps.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

    if len(steps) >= 2 and len(grad_norm) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, grad_norm, color="mediumseagreen", linewidth=1.0, label="Grad Norm")
        ax.axhline(y=1.0, color="tomato", linestyle="--", linewidth=1.0, label="Clip threshold (1.0)")
        ax.set_title("Gradient Norm per Step"); ax.set_xlabel("Step"); ax.set_ylabel("Grad Norm")
        ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "grad_norm_steps.png"), dpi=150, bbox_inches="tight")
        plt.close("all")


def plot_epoch_metrics(epoch_history: dict, plot_dir: str) -> None:
    epochs     = epoch_history.get("epochs",      [])
    train_loss = epoch_history.get("train_loss",  [])
    val_loss   = epoch_history.get("val_loss",    [])
    val_acc    = epoch_history.get("val_accuracy",[])
    val_f1     = epoch_history.get("val_f1",      [])
    epoch_dir  = os.path.join(plot_dir, "epoch_plots")

    if len(epochs) < 1:
        return

    if len(train_loss) >= 1 and len(val_loss) >= 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, color="steelblue", marker="o", linewidth=1.5, label="Train Loss")
        ax.plot(epochs, val_loss,   color="tomato",    marker="o", linewidth=1.5, label="Val Loss")
        best_idx = val_loss.index(min(val_loss))
        ax.plot(epochs[best_idx], val_loss[best_idx], marker="*", color="limegreen",
                markersize=14, zorder=5, label=f"Best (epoch {epochs[best_idx]})")
        ax.set_title("Train vs Val Loss per Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, "train_val_loss.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

    if len(val_acc) >= 1 and len(val_f1) >= 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, val_acc, color="steelblue",  marker="o", linewidth=1.5, label="Val Accuracy")
        ax.plot(epochs, val_f1,  color="darkorange",  marker="o", linewidth=1.5, label="Val F1 (macro)")
        ax.axhline(y=0.5, color="gray",          linestyle="--", linewidth=0.9, label="Random (0.5)")
        ax.axhline(y=0.8, color="mediumseagreen", linestyle="--", linewidth=0.9, label="Good (0.8)")
        ax.axhline(y=0.9, color="gold",          linestyle="--", linewidth=0.9, label="Excellent (0.9)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Validation Metrics per Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, "val_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close("all")


def plot_confusion_matrix(tp: int, tn: int, fp: int, fn: int, epoch: int, plot_dir: str) -> None:
    total = tn + fp + fn + tp
    if total == 0:
        return
    epoch_dir = os.path.join(plot_dir, "epoch_plots")
    matrix   = [[tn, fp], [fn, tp]]
    labels   = ["Benign", "Malware"]
    flat_max = max(tn, fp, fn, tp) or 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=flat_max)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — Epoch {epoch}")

    for i in range(2):
        for j in range(2):
            val = matrix[i][j]
            pct = 100.0 * val / total
            text_color = "white" if val > flat_max * 0.5 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)",
                    ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
