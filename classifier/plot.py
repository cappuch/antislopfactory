import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def plot(log_path: str = "data/safety_classifier.json"):
    log = json.loads(Path(log_path).read_text())
    h = log["history"]
    epochs = list(range(1, len(h["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(epochs, h["train_loss"], label="train")
    if h["val_loss"]:
        ax.plot(epochs, h["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, h["train_acc"], label="train")
    if h["val_acc"]:
        ax.plot(epochs, h["val_acc"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    pc = log["test_per_class"]
    cats = sorted(pc.keys(), key=lambda c: pc[c]["accuracy"])
    accs = [pc[c]["accuracy"] for c in cats]
    counts = [pc[c]["total"] for c in cats]
    bars = ax.barh(cats, accs)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=8)
    ax.set_xlabel("accuracy")
    ax.set_title(f"Test per-class (overall={log['test_accuracy']:.3f})")
    ax.set_xlim(0, 1.15)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"{log['train_samples']} train / "
                 f"{log['val_samples']} val / {log['test_samples']} test",
                 fontsize=11)
    fig.tight_layout()

    out = Path(log_path).with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"saved to {out}")
    plt.show()


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else "data/safety_classifier.json")
