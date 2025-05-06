import cv2
import numpy as np
import os


class Preprocessor:

    def __init__():
        pass

    def deskew_image(self, img):
        coords = np.column_stack(np.where(img > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)

    
    def denoise_image(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)


    def binary_thresholding(self, img):
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.)


    def preprocess_image(self, img_path: str, target_size: int = 1024) -> np.ndarray:
        """
        Combined preprocess steps for the input image for YOLOv8:
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
