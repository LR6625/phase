"""OBB training progress monitor with best-epoch tracking.

Tracks:
- mAP50
- mAP50-95
- fitness (0.1*mAP50 + 0.9*mAP50-95)

Examples:
    python watch_obb_progress.py
    python watch_obb_progress.py --run-dir runs/obb/detect_obb_combined_v1 --max-epochs 200 --refresh-secs 2
"""
import argparse
import csv
import time
from pathlib import Path

BAR_WIDTH = 40


def get_val(row, key_substring):
    for k, v in row.items():
        if key_substring.lower() in k.strip().lower():
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def load_rows(csv_path: Path):
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows
    except Exception:
        return []


def fmt(x, f=".4f"):
    return format(x, f) if x is not None else "n/a"


def make_bar(epoch_now, epoch_max):
    epoch_now = max(0, min(epoch_now, epoch_max))
    ratio = epoch_now / float(epoch_max)
    filled = int(BAR_WIDTH * ratio)
    return "[" + ("#" * filled) + ("-" * (BAR_WIDTH - filled)) + f"] {epoch_now}/{epoch_max}  {ratio*100:5.1f}%"


def make_fraction_bar(frac: float) -> str:
    frac = max(0.0, min(frac, 1.0))
    filled = int(BAR_WIDTH * frac)
    return "[" + ("#" * filled) + ("-" * (BAR_WIDTH - filled)) + f"] {frac*100:5.1f}%"


def summarize_best(rows):
    best_map50 = (-1.0, -1)
    best_map5095 = (-1.0, -1)
    best_fitness = (-1.0, -1)

    for r in rows:
        ep0 = get_val(r, "epoch")
        if ep0 is None:
            continue
        ep = int(ep0) + 1

        m50 = get_val(r, "metrics/mAP50(B)")
        m95 = get_val(r, "metrics/mAP50-95(B)")
        if m50 is None:
            m50 = 0.0
        if m95 is None:
            m95 = 0.0
        fit = 0.1 * m50 + 0.9 * m95

        if m50 > best_map50[0]:
            best_map50 = (m50, ep)
        if m95 > best_map5095[0]:
            best_map5095 = (m95, ep)
        if fit > best_fitness[0]:
            best_fitness = (fit, ep)

    return best_map50, best_map5095, best_fitness


def find_latest_obb_run() -> Path:
    obb_root = Path("runs/obb")
    candidates = [p for p in obb_root.glob("*") if p.is_dir()]
    if not candidates:
        return Path("runs/obb/detect_obb_run1")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch OBB training progress from Ultralytics results.csv")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory under runs/obb (auto-detect latest if omitted)")
    parser.add_argument("--max-epochs", type=int, default=200, help="Expected max epochs for progress bar")
    parser.add_argument("--refresh-secs", type=float, default=2.0, help="Refresh interval in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_obb_run()
    results_csv = run_dir / "results.csv"

    print(f"  OBB Training Monitor: {results_csv}")
    print(f"  Refresh: {args.refresh_secs}s | Max epochs: {args.max_epochs} | Ctrl+C to exit\n")

    stagnant_polls = 0
    last_row_count = 0
    last_epoch_change_time = time.time()
    last_epoch_seen = 0
    avg_epoch_seconds = None
    waiting_line_active = False
    last_wait_bucket = -1

    try:
        while True:
            if not results_csv.exists():
                msg = "  Waiting for results.csv to appear..."
                print(msg, end="\r", flush=True)
                waiting_line_active = True
                time.sleep(args.refresh_secs)
                continue

            rows = load_rows(results_csv)
            if not rows:
                msg = "  results.csv exists, waiting for first epoch row..."
                print(msg, end="\r", flush=True)
                waiting_line_active = True
                time.sleep(args.refresh_secs)
                continue

            if len(rows) == last_row_count:
                stagnant_polls += 1
            else:
                stagnant_polls = 0
                last_row_count = len(rows)

            latest = rows[-1]
            epoch0 = get_val(latest, "epoch")
            epoch = int(epoch0) + 1 if epoch0 is not None else len(rows)

            times = []
            for r in rows:
                t = get_val(r, "time")
                if t is not None:
                    times.append(float(t))
            if len(times) >= 2:
                deltas = []
                for i in range(1, len(times)):
                    d = times[i] - times[i - 1]
                    if d > 0:
                        deltas.append(d)
                if deltas:
                    window = deltas[-8:]
                    avg_epoch_seconds = sum(window) / len(window)

            box_loss = get_val(latest, "train/box_loss")
            cls_loss = get_val(latest, "train/cls_loss")
            precision = get_val(latest, "metrics/precision(B)")
            recall = get_val(latest, "metrics/recall(B)")
            map50 = get_val(latest, "metrics/mAP50(B)")
            map5095 = get_val(latest, "metrics/mAP50-95(B)")
            if map50 is None:
                map50 = 0.0
            if map5095 is None:
                map5095 = 0.0
            fitness = 0.1 * map50 + 0.9 * map5095

            best_map50, best_map5095, best_fitness = summarize_best(rows)

            if stagnant_polls == 0:
                if waiting_line_active:
                    print(" " * 140, end="\r", flush=True)
                    waiting_line_active = False

                if epoch != last_epoch_seen:
                    last_epoch_change_time = time.time()
                    last_epoch_seen = epoch
                    last_wait_bucket = -1

                print(f"  Progress         {make_bar(epoch, args.max_epochs)}")
                print(f"  Epoch            {epoch}/{args.max_epochs}")
                print(f"  Loss             box={fmt(box_loss)}   cls={fmt(cls_loss)}")
                print(
                    f"  Current metrics  P={fmt(precision)}   R={fmt(recall)}   mAP50={fmt(map50)}   mAP50-95={fmt(map5095)}   fitness={fmt(fitness)}"
                )
                print(f"  Best mAP50       {best_map50[0]:.4f} @ epoch {best_map50[1]}")
                print(f"  Best mAP50-95    {best_map5095[0]:.4f} @ epoch {best_map5095[1]}")
                print(f"  Best fitness     {best_fitness[0]:.4f} @ epoch {best_fitness[1]}")
                if avg_epoch_seconds is not None:
                    print(f"  Avg epoch time   ~{avg_epoch_seconds/60.0:.2f} min/epoch")
                print("")
            else:
                if avg_epoch_seconds is not None and avg_epoch_seconds > 1e-6:
                    elapsed = max(0.0, time.time() - last_epoch_change_time)
                    est_frac = min(0.99, elapsed / avg_epoch_seconds)
                    bucket = int(est_frac * 10)  # 10% milestones
                    if bucket != last_wait_bucket:
                        last_wait_bucket = bucket
                        msg = (
                            f"  In-epoch status  epoch {epoch} complete; waiting on epoch {epoch + 1} | "
                            f"est progress {make_fraction_bar(est_frac)} | {elapsed/60.0:.2f}m elapsed"
                        )
                        print(msg)
                else:
                    if stagnant_polls % 30 == 0:
                        msg = (
                            f"  In-epoch status  epoch {epoch} complete; waiting on epoch {epoch + 1} | "
                            "estimating..."
                        )
                        print(msg)

            if epoch >= args.max_epochs:
                if waiting_line_active:
                    print(" " * 140, end="\r", flush=True)
                print("\n  Reached MAX_EPOCHS.")
                break

            time.sleep(args.refresh_secs)

    except KeyboardInterrupt:
        print("\n  Monitor stopped.")


if __name__ == "__main__":
    main()
