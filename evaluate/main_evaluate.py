from ultralytics import YOLO
from doctr_utils import load_doctr_model
from evaluate import run_inference_on_folder

def main():
    # Paths
    test_images_dir = "dataset_25_obb/test/images"
    output_json = "outputs/test_predictions.json"

    # Load trained YOLOv8 model
    yolo_model = YOLO("runs/detect/train/weights/best.pt")  # or your best model path

    # Load OCR model
    doctr_model = load_doctr_model()

    # Run evaluation
    run_inference_on_folder(yolo_model, doctr_model, test_images_dir, output_json)

if __name__ == "__main__":
    main()

