"""
Validate the trained OBB model and print key metrics.
Run after train_obb.py completes.
"""
from ultralytics import YOLO


def main() -> None:
    model = YOLO("runs/obb/detect_obb_run1/weights/best.pt")
    results = model.val(data="dataset_obb.yaml", split="val", imgsz=640, device="cpu")

    p = float(results.box.p[0])
    r = float(results.box.r[0])
    map50 = float(results.box.map50)
    map5095 = float(results.box.map)

    print(f"OBB_P={p:.6f}")
    print(f"OBB_R={r:.6f}")
    print(f"OBB_mAP50={map50:.6f}")
    print(f"OBB_mAP50_95={map5095:.6f}")

    # Visual previews on val set
    model.predict(
        source="yolo_dataset/images/val",
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        project="runs/obb",
        name="best_guesses_obb_run1",
        exist_ok=True,
    )
    print("Preview images saved to runs/obb/best_guesses_obb/")


if __name__ == "__main__":
    main()
