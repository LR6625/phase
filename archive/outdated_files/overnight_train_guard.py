import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep OBB training alive overnight via auto-resume")
    parser.add_argument("--run-dir", type=str, default="runs/obb/detect_obb_combined_v1")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--check-secs", type=float, default=30.0)
    parser.add_argument("--python-exe", type=str, default="python")
    parser.add_argument("--log-path", type=str, default=None)
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(log_path: Path, message: str) -> None:
    line = f"[{now()}] {message}\n"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        # If file is locked, still keep console heartbeat alive.
        pass
    print(line, end="", flush=True)


def latest_epoch(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    try:
        with open(results_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return 0
        by_epoch = {}
        for row in rows:
            try:
                ep = int(float(row.get("epoch", ""))) + 1
            except Exception:
                continue
            by_epoch[ep] = row
        return max(by_epoch.keys()) if by_epoch else 0
    except Exception:
        return 0


def trainer_running(run_dir_str: str) -> bool:
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'detect_obb_combined_v1|last.pt|dataset_obb_combined' } | "
        "Measure-Object | Select-Object -ExpandProperty Count"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    try:
        count = int((out.stdout or "0").strip().splitlines()[-1])
    except Exception:
        count = 0
    return count > 0


def start_resume_trainer(python_exe: str, run_dir: Path) -> None:
    ckpt = run_dir / "weights" / "last.pt"
    script = (
        "from ultralytics import YOLO; "
        f"YOLO(r'{str(ckpt)}').train(resume=True)"
    )
    subprocess.Popen([python_exe, "-c", script])


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    results_csv = run_dir / "results.csv"
    log_path = Path(args.log_path) if args.log_path else (run_dir / "overnight_guard_watchdog.log")

    log_line(log_path, "Guard started")

    while True:
        ep = latest_epoch(results_csv)
        if ep >= args.max_epochs:
            log_line(log_path, f"Training complete at epoch {ep}. Guard exiting.")
            break

        alive = trainer_running(str(run_dir))
        if not alive:
            log_line(log_path, f"No trainer process detected at epoch {ep}. Starting resume trainer.")
            start_resume_trainer(args.python_exe, run_dir)
            time.sleep(5)
        else:
            log_line(log_path, f"Trainer alive at epoch {ep}.")

        time.sleep(args.check_secs)


if __name__ == "__main__":
    main()
