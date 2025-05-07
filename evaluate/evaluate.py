import os
import json
from tqdm import tqdm
import numpy as np
from glob import glob
from yolo_utils import predict_yolo
from doctr_utils import load_doctr_model, extract_text_from_crops

# Mapping of class index to field names
CLASS_MAP = {
    0: "charges",
    1: "mawb_prefix",
    2: "mawb_serial"
}

def run_inference_on_folder(yolo_model, doctr_model, image_folder: str, output_json_path: str):
    """
    Runs YOLOv8 OBB detection + DocTR OCR on a folder of test images.
    Saves structured JSON with predicted fields per document.
    """
    results = {}
    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))

    for img_path in tqdm(image_paths, desc="Evaluating"):
        image_name = os.path.basename(img_path)

        # Run YOLO OBB prediction
        obb_results = predict_yolo(yolo_model, img_path)
        bboxes = obb_results.xyxyxyxy.cpu().numpy()
        classes = obb_results.cls.cpu().numpy()

        # Run OCR on the predicted bounding boxes
        texts = extract_text_from_crops(img_path, bboxes, doctr_model)

        # Build structured result
        result = {}
        for cls_id, text in zip(classes, texts):
            field_name = CLASS_MAP.get(int(cls_id), f"class_{cls_id}")
            result[field_name] = text

        results[image_name] = result

    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete. Output saved to: {output_json_path}")


