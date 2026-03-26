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
class DetectionFrame:
    """Frame with detections for sending to frontend"""
    frame_id: int
    timestamp: float
    detections: List[Detection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "detections",
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detection_count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string for WebRTC data channel"""
        return json.dumps(self.to_dict())
    
    @staticmethod
    def from_yolo_detections(
        frame_id: int,
        timestamp: float,
        yolo_detections: List[Any],  # List[YoloDetection]
    ) -> "DetectionFrame":
        """Convert YoloDetector results to DetectionFrame"""
        detections = []
        for yolo_det in yolo_detections:
            x1, y1, x2, y2 = yolo_det.xyxy
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
        )
