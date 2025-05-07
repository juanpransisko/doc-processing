from dataclasses import dataclass
from typing import List, Dict, Optional
import re

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def centroid(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def intersects(self, other: 'BBox') -> bool:
        return not (
            self.x2 < other.x1 or self.x1 > other.x2 or
            self.y2 < other.y1 or self.y1 > other.y2
        )

@dataclass
class Prediction:
    class_id: int
    bbox: BBox
    value: str
    confidence: float

@dataclass
class AWBFieldResult:
    value: Optional[str]
    confidence: float
    source_region: str
    class_id: int


class AWBResolver:
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height

    def categorize_region(self, bbox: BBox) -> str:
        x, y = bbox.centroid()
        if x < self.image_width / 2 and y < self.image_height / 2:
            return 'top_left'
        elif x >= self.image_width / 2 and y < self.image_height / 2:
            return 'top_right'
        else:
            return 'bottom_right'

    def resolve_by_region(self, 
                            predictions: List[Prediction]
        ) -> Dict[str, Dict[str, AWBFieldResult]]:
        result = {'awb_prefix': {}, 'awb_serial': {}}
        for pred in predictions:
            region = self.categorize_region(pred.bbox)

            if pred.class_id == 1 and re.fullmatch(r"\d{3}", pred.value):
                prev = result['awb_prefix'].get(region)
                if not prev or pred.confidence > prev.confidence:
                    result['awb_prefix'][region] = AWBFieldResult(pred.value, pred.confidence, region, pred.class_id)

            elif pred.class_id == 2 and re.fullmatch(r"\d{8}", pred.value):
                prev = result['awb_serial'].get(region)
                if not prev or pred.confidence > prev.confidence:
                    result['awb_serial'][region] = AWBFieldResult(pred.value, pred.confidence, region, pred.class_id)

        return result

    def resolve_consistent(self, region_result: Dict[str, Dict[str, AWBFieldResult]]) -> Dict[str, Optional[str]]:
        """Return final result if all zones agree, else None or fallback logic"""
        prefixes = list({v.value for v in region_result['awb_prefix'].values()})
        serials = list({v.value for v in region_result['awb_serial'].values()})

        consistent = len(prefixes) == 1 and len(serials) == 1

        return {
            "prefix": prefixes[0] if consistent else None,
            "serial": serials[0] if consistent else None,
            "consistent": consistent
        }

# Convert raw list to Prediction objects
raw_preds = [
    {'class': 1, 'bbox': [50, 50, 100, 100], 'value': '123', 'confidence': 0.95},
    {'class': 2, 'bbox': [200, 200, 300, 300], 'value': '12345678', 'confidence': 0.92},
    # etc.
]
predictions = [Prediction(p['class'], BBox(*p['bbox']), p['value'], p['confidence']) for p in raw_preds]

resolver = AWBResolver(image_width=600, image_height=400)
region_based = resolver.resolve_by_region(predictions)
final_result = resolver.resolve_consistent(region_based)

print(final_result)
# Expected output: {'prefix': '123', 'serial': '12345678', 'consistent': True}


