import argparse
import json
import time
from pathlib import Path


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(float(seconds), 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_bar(percent: float, width: int = 40) -> str:
    pct = max(0.0, min(100.0, float(percent)))
    fill = int(round((pct / 100.0) * width))
    return "[" + ("#" * fill) + ("-" * (width - fill)) + "]"


def clear_screen():
    print("\x1b[2J\x1b[H", end="")


def main():
    parser = argparse.ArgumentParser(description="Monitor progress of segmented DB v2 build.")
    parser.add_argument(
        "--progress-file",
        default="scryfall_segmented_tophalf_db_v2.json.progress.json",
        help="Progress JSON written by build script",
    )
    parser.add_argument("--refresh", type=float, default=5.0, help="Refresh interval seconds")
    args = parser.parse_args()

    progress_path = Path(args.progress_file)
    if not progress_path.exists():
        raise FileNotFoundError(f"Progress file not found: {progress_path}")

    while True:
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception as exc:
            clear_screen()
            print(f"Could not read progress file: {exc}")
            time.sleep(args.refresh)
            continue

        status = payload.get("status", "unknown")
        percent = float(payload.get("percent", 0.0))
        total = int(payload.get("total_candidates", 0))
        scanned = int(payload.get("scanned", 0))
        built = int(payload.get("built", 0))
        skipped = int(payload.get("skipped", 0))
        errors = int(payload.get("errors", 0))
        eta = payload.get("eta_seconds")

        clear_screen()
        print("Segmented DB v2 Build Monitor")
        print("=" * 40)
        print(f"Status   : {status}")
        print(f"Progress : {make_bar(percent)} {percent:6.2f}%")
        print(f"Scanned  : {scanned}/{total}")
        print(f"Built    : {built}")
        print(f"Skipped  : {skipped}")
        print(f"Errors   : {errors}")
        print(f"ETA      : {format_seconds(eta)}")
        print(f"Updated  : {format_seconds(time.time() - float(payload.get('updated_at', time.time())))} ago")
        print(f"Output   : {payload.get('output', 'n/a')}")

        if status == "done":
            break

        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
