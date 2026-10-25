from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n-obb.pt")
    model.train(
        data="dataset_obb_combined.yaml",
        epochs=200,
        patience=50,
        imgsz=640,
        batch=2,
        workers=2,
        device="cpu",
        degrees=50,
        shear=4.0,
        perspective=0.0005,
        translate=0.10,
        scale=0.30,
        mosaic=0.30,
        name="detect_obb_combined_v1",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
