from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="dataset.yaml",
        imgsz=640,
        epochs=40,
        batch=2,
        workers=2,
        device="cpu",
        project="runs",
        name="detect_baseline_cpu",
    )


if __name__ == "__main__":
    main()
