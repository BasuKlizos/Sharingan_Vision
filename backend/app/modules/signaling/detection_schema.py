"""
Detection Data Schemas for WebRTC Data Channel Communication
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class Detection:
    """Individual object detection"""
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 2),
            "bounding_box": self.bounding_box.to_dict(),
        }


@dataclass
class CropOffset:
    """Crop region offset for coordinate transformation"""
    x_offset: int = 0
    y_offset: int = 0
    original_width: int = 0
    original_height: int = 0
    cropped_width: int = 0
    cropped_height: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class DetectionFrame:
    """Frame with detections for sending to frontend"""
    frame_id: int
    timestamp: float
    detections: List[Detection]
    crop_offset: CropOffset = None
    
    def __post_init__(self):
        if self.crop_offset is None:
            self.crop_offset = CropOffset()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "detections",
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detection_count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "crop_offset": self.crop_offset.to_dict() if self.crop_offset else None,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string for WebRTC data channel"""
        return json.dumps(self.to_dict())
    
    @staticmethod
    def from_yolo_detections(
        frame_id: int,
        timestamp: float,
        yolo_detections: List[Any],  # List[YoloDetection]
        crop_offset: "CropOffset" = None,
    ) -> "DetectionFrame":
        """Convert YoloDetector results to DetectionFrame
        
        Args:
            frame_id: Frame identifier
            timestamp: Frame timestamp
            yolo_detections: List of YOLO detection results
            crop_offset: CropOffset object with x/y offsets if frame was cropped
        """
        detections = []
        for yolo_det in yolo_detections:
            x1, y1, x2, y2 = yolo_det.xyxy
            
            # Apply crop offset to convert back to original frame coordinates
            if crop_offset and (crop_offset.x_offset > 0 or crop_offset.y_offset > 0):
                x1 += crop_offset.x_offset
                y1 += crop_offset.y_offset
                x2 += crop_offset.x_offset
                y2 += crop_offset.y_offset
            
            detection = Detection(
                class_id=yolo_det.class_id,
                class_name=yolo_det.class_name,
                confidence=yolo_det.confidence,
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            )
            detections.append(detection)
        
        return DetectionFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            crop_offset=crop_offset or CropOffset(),
        )
