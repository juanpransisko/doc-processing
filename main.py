from model.ocr_yolo_model import train_yolo_model, save_data_yaml, predict_yolo
from model.doctr_model import load_doctr_model, extract_text_from_crops
import os


def main():
    # Dataset config
    config = {
        "path": "dataset_25_obb",
        "train": "train",
        "val": "valid",
        "test": "test",
        "nc": 3,
        "names": ["charges", "mawb_prefix", "mawb_serial"],
    }
    save_data_yaml(config)

    # Train YOLOv8 OBB
    model = train_yolo_model(
        "dataset_25_obb/data.yaml", model_arch="yolov8n-obb.pt", epochs=50
    )

    # Load OCR model
    ocr_model = load_doctr_model()

    # Inference example
    sample_image = "dataset_25_obb/test/images/example.jpg"
    obb_results = predict_yolo(model, sample_image)
    bboxes = obb_results.xyxyxyxy.cpu().numpy()
    classes = obb_results.cls.cpu().numpy()

    # Extract text from predicted regions
    texts = extract_text_from_crops(sample_image, bboxes, ocr_model)

    for cls, txt in zip(classes, texts):
        print(f"Detected Class {int(cls)} ({config['names'][int(cls)]}): {txt}")


if __name__ == "__main__":
    main()
