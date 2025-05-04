import cv2
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from typing import List


def load_doctr_model():
    """Load a lightweight OCR model."""
    return ocr_predictor(
        det_arch="db_resnet50", reco_arch="crnn_mobilenet_v3_small", pretrained=True
    )


def extract_text_from_crops(image_path: str, bboxes: np.ndarray, model) -> List[str]:
    """
    Extract text from oriented bounding box regions using DocTR.
    Assumes bboxes are in 8-point OBB format.
    """
    img = cv2.imread(image_path)
    texts = []
    for box in bboxes:
        pts = np.array(box, dtype=np.int32).reshape(4, 2)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        crop = img[y : y + h, x : x + w]

        if crop.size == 0:
            texts.append("")
            continue

        crop_resized = cv2.resize(crop, (128, 32), interpolation=cv2.INTER_CUBIC)
        doc = DocumentFile.from_images(crop_resized)
        result = model(doc)
        text = result.pages[0].extract_text()
        texts.append(text.strip() if text else "")
    return texts
