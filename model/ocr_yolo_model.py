import cv2
import numpy as np
import os
from ultralytics import YOLO
import yaml
from typing import List


def save_data_yaml(config: dict, path="dataset_25_obb/data.yaml"):
    """Save dataset YAML file for YOLOv8 training."""
    with open(path, "w") as f:
        yaml.dump(config, f)


def train_yolo_model(yaml_path: str, model_arch="yolov8n-obb.pt", epochs=50, imgsz=1024):
    model = YOLO(model_arch)
    model.train(data=yaml_path, epochs=epochs, imgsz=imgsz)
    return model


def predict_yolo(model: YOLO, image_path: str):
    return model.predict(image_path, save=False)[0].obb
