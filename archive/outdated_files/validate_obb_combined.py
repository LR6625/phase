from ultralytics import YOLO


def main() -> None:
    model = YOLO("runs/obb/detect_obb_combined_v1/weights/best.pt")

    test_results = model.val(
        data="dataset_obb_combined.yaml",
        split="test",
        imgsz=640,
        device="cpu",
    )
    val_results = model.val(
        data="dataset_obb_combined.yaml",
        split="val",
        imgsz=640,
        device="cpu",
    )

    print(f"TEST_OBB_P={float(test_results.box.mp):.6f}")
    print(f"TEST_OBB_R={float(test_results.box.mr):.6f}")
    print(f"TEST_OBB_mAP50={float(test_results.box.map50):.6f}")
    print(f"TEST_OBB_mAP50_95={float(test_results.box.map):.6f}")

    print(f"VAL_OBB_P={float(val_results.box.mp):.6f}")
    print(f"VAL_OBB_R={float(val_results.box.mr):.6f}")
    print(f"VAL_OBB_mAP50={float(val_results.box.map50):.6f}")
    print(f"VAL_OBB_mAP50_95={float(val_results.box.map):.6f}")

    model.predict(
        source="yolo_obb_combined/images/test",
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        project="runs/obb",
        name="preview_obb_combined_test",
        exist_ok=True,
    )
    print("Preview images saved to runs/obb/preview_obb_combined_test")


if __name__ == "__main__":
    main()
