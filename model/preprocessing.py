import cv2
import numpy as np
import os
from ultralytics import YOLO
import yaml
from typing import List


def preprocess_image(img_path: str, target_size: int = 1024) -> np.ndarray:
    """
    Preprocess the input image for YOLOv8:
    - Resize
    - Deskew (if applicable)
    - Histogram normalization (contrast enhancement)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Histogram equalization per channel
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Resize
    img_resized = cv2.resize(
        img_eq, (target_size, target_size), interpolation=cv2.INTER_AREA
    )
    return img_resized
