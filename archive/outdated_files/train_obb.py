"""
Train YOLOv8n-OBB (oriented bounding boxes) on the card dataset.
Base model: yolov8n-obb.pt (auto-downloaded on first run).
OBB labels must exist in yolo_dataset/labels_obb/{train,val,test}.
"""
from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n-obb.pt")  # downloads ~6 MB on first run
    model.train(
        data="dataset_obb.yaml",
        epochs=200,
        patience=50,          # auto-stop if no improvement for 50 epochs
        imgsz=640,
        batch=2,
        workers=2,
        device="cpu",
        # Rotation augmentation — critical for rotated cards
        degrees=50,
        shear=4.0,
        perspective=0.0005,
        translate=0.10,
        scale=0.30,
        mosaic=0.30,
        name="detect_obb_run1",
        exist_ok=True,
    )
    print("OBB training complete. Run validate_obb.py to get final metrics.")


if __name__ == "__main__":
    main()
