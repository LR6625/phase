import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live OBB training dashboard")
    parser.add_argument("--run-dir", type=str, default="runs/obb/detect_obb_combined_v1")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--refresh-secs", type=float, default=20.0)
    parser.add_argument("--save-graphs", action="store_true", help="Save dashboard snapshots to disk")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save graph images")
    return parser.parse_args()


def read_rows(results_csv: Path):
    if not results_csv.exists():
        return []

    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Keep last occurrence per epoch in case resumes duplicate rows.
    by_epoch = {}
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", ""))) + 1
        except Exception:
            continue
        by_epoch[epoch] = row

    epochs = sorted(by_epoch.keys())
    return [by_epoch[e] for e in epochs]


def fval(row, key):
    try:
        return float(row.get(key, ""))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    results_csv = run_dir / "results.csv"
    save_dir = Path(args.save_dir) if args.save_dir else (run_dir / "dashboard_snapshots")
    if args.save_graphs:
        save_dir.mkdir(parents=True, exist_ok=True)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    fig.canvas.manager.set_window_title("OBB Training Dashboard")
    last_saved_epoch = -1

    while True:
        rows = read_rows(results_csv)
        ax1.clear()
        ax2.clear()

        if not rows:
            ax1.text(0.5, 0.5, f"Waiting for {results_csv}", ha="center", va="center", fontsize=12)
            ax1.set_axis_off()
            ax2.set_axis_off()
            plt.pause(0.1)
            time.sleep(args.refresh_secs)
            continue

        epochs = []
        map50 = []
        map5095 = []
        precision = []
        recall = []
        train_box = []

        for row in rows:
            e = int(float(row["epoch"])) + 1
            epochs.append(e)
            map50.append(fval(row, "metrics/mAP50(B)") or 0.0)
            map5095.append(fval(row, "metrics/mAP50-95(B)") or 0.0)
            precision.append(fval(row, "metrics/precision(B)") or 0.0)
            recall.append(fval(row, "metrics/recall(B)") or 0.0)
            train_box.append(fval(row, "train/box_loss") or 0.0)

        best_m50 = max(map50) if map50 else 0.0
        best_m95 = max(map5095) if map5095 else 0.0
        best_m50_ep = epochs[map50.index(best_m50)] if map50 else 0
        best_m95_ep = epochs[map5095.index(best_m95)] if map5095 else 0
        current_epoch = epochs[-1]

        progress = min(1.0, max(0.0, current_epoch / float(max(1, args.max_epochs))))
        fig.suptitle(
            f"Run: {run_dir} | Epoch {current_epoch}/{args.max_epochs} ({progress*100:.1f}%)\n"
            f"Best mAP50: {best_m50:.4f} @ {best_m50_ep} | Best mAP50-95: {best_m95:.4f} @ {best_m95_ep}",
            fontsize=11,
        )

        ax1.plot(epochs, map50, label="mAP50", linewidth=2)
        ax1.plot(epochs, map5095, label="mAP50-95", linewidth=2)
        ax1.plot(epochs, precision, label="Precision", linestyle="--", alpha=0.8)
        ax1.plot(epochs, recall, label="Recall", linestyle="--", alpha=0.8)
        ax1.set_ylabel("Metric")
        ax1.set_ylim(0.0, 1.02)
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="lower right")

        ax2.plot(epochs, train_box, label="train/box_loss", color="tab:red", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if args.save_graphs:
            latest_path = save_dir / "latest.png"
            fig.savefig(latest_path, dpi=140, bbox_inches="tight")

            if current_epoch != last_saved_epoch:
                epoch_path = save_dir / f"epoch_{current_epoch:03d}.png"
                fig.savefig(epoch_path, dpi=140, bbox_inches="tight")
                last_saved_epoch = current_epoch

            if current_epoch >= args.max_epochs:
                final_path = save_dir / "final.png"
                fig.savefig(final_path, dpi=160, bbox_inches="tight")

        plt.pause(0.1)

        if current_epoch >= args.max_epochs:
            print(f"Training reached epoch {current_epoch}. Final dashboard saved to {save_dir}", flush=True)
            break

        time.sleep(args.refresh_secs)


if __name__ == "__main__":
    main()
