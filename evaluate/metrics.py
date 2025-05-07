import json
from difflib import SequenceMatcher
from typing import Dict, Tuple
from collections import defaultdict
import numpy as np


def compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate."""
    if not gt:
        return 1.0 if pred else 0.0
    return 1 - SequenceMatcher(None, pred, gt).ratio()


def compute_wer(pred: str, gt: str) -> float:
    """Word Error Rate."""
    pred_words = pred.split()
    gt_words = gt.split()
    sm = SequenceMatcher(None, pred_words, gt_words)
    return 1 - sm.ratio()


def evaluate_json(pred_path: str, gt_path: str) -> Dict[str, float]:
    with open(pred_path) as f:
        pred_data = json.load(f)

    with open(gt_path) as f:
        gt_data = json.load(f)

    total_fields = 0
    correct_fields = 0
    cer_list = []
    wer_list = []

    field_totals = defaultdict(int)
    field_correct = defaultdict(int)

    for image_id, gt_fields in gt_data.items():
        pred_fields = pred_data.get(image_id, {})

        for field_name, gt_value in gt_fields.items():
            pred_value = pred_fields.get(field_name, "")
            total_fields += 1
            field_totals[field_name] += 1

            if pred_value.strip() == gt_value.strip():
                correct_fields += 1
                field_correct[field_name] += 1

            cer_list.append(compute_cer(pred_value, gt_value))
            wer_list.append(compute_wer(pred_value, gt_value))

    # Aggregate results
    accuracy = correct_fields / total_fields
    cer = np.mean(cer_list)
    wer = np.mean(wer_list)

    per_field_accuracy = {
        field: field_correct[field] / total if total > 0 else 0
        for field, total in field_totals.items()
    }

    return {
        "overall_accuracy": round(accuracy * 100, 2),
        "character_error_rate": round(cer * 100, 2),
        "word_error_rate": round(wer * 100, 2),
        "per_field_accuracy": {k: round(v * 100, 2) for k, v in per_field_accuracy.items()},
        "total_fields_evaluated": total_fields
    }

