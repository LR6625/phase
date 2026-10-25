from ultralytics import YOLO


def main() -> None:
    model = YOLO("runs/detect/runs/detect_manual301_quick/weights/best.pt")
    results = model.val(data="dataset.yaml", split="val", imgsz=640, device="cpu")

    p = float(results.box.p[0])
    r = float(results.box.r[0])
    map50 = float(results.box.map50)
    map5095 = float(results.box.map)

    print(f"FINAL_P={p:.6f}")
    print(f"FINAL_R={r:.6f}")
    print(f"FINAL_mAP50={map50:.6f}")
    print(f"FINAL_mAP50_95={map5095:.6f}")


if __name__ == "__main__":
    main()