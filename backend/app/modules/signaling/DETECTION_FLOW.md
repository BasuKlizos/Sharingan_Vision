# Detection Data Flow Documentation

## Overview
This backend system processes video frames with YOLO detection and sends structured detection results to the frontend in real-time using WebRTC data channels. The frontend (in separate repo) receives and visualizes detected objects with bounding boxes.

## Architecture

```
BACKEND FLOW:
┌─────────────────────────────────────────────────────┐
│ 1. WebRTC Peer Connection Established               │
├─────────────────────────────────────────────────────┤
│ 2. Detection Data Channel Created (outbound)        │
├─────────────────────────────────────────────────────┤
│ 3. Video Frames Received from Client               │
├─────────────────────────────────────────────────────┤
│ 4. YoloDetector.detect(frame)                       │
│    → List[YoloDetection]                            │
├─────────────────────────────────────────────────────┤
│ 5. Convert to DetectionFrame (structured)           │
├─────────────────────────────────────────────────────┤
│ 6. Send JSON via Data Channel                       │
└─────────────────────────────────────────────────────┘
         ↓
    [FRONTEND RECEIVES & RENDERS]
```

## Data Structures

### Backend Structures

#### YoloDetection (from yolo_detector.py)
```python
@dataclass(frozen=True)
class YoloDetection:
    class_id: int                    # e.g., 0 for "person"
    class_name: str                 # e.g., "person"
    confidence: float               # 0.0-1.0
    xyxy: Tuple[int, int, int, int] # (x1, y1, x2, y2) bounding box
```

#### BoundingBox (detection_schema.py)
```python
@dataclass
class BoundingBox:
    x1: int  # top-left x
    y1: int  # top-left y
    x2: int  # bottom-right x
    y2: int  # bottom-right y
```

#### Detection (detection_schema.py)
```python
@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
```

#### DetectionFrame (detection_schema.py)
```python
@dataclass
class DetectionFrame:
    frame_id: int           # Sequential frame counter
    timestamp: float        # time.time()
    detections: List[Detection]
    
    # Methods:
    to_dict()          # Convert to dictionary
    to_json()          # Convert to JSON string for sending
    from_yolo_detections()  # Create from YoloDetector output
```

### JSON Format Sent to Frontend

```json
{
  "type": "detections",
  "frame_id": 42,
  "timestamp": 1711427195.3842,
  "detection_count": 3,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.92,
      "bounding_box": {
        "x1": 100,
        "y1": 50,
        "x2": 300,
        "y2": 400
      }
    },
    {
      "class_id": 1,
      "class_name": "car",
      "confidence": 0.88,
      "bounding_box": {
        "x1": 350,
        "y1": 100,
        "x2": 600,
        "y2": 300
      }
    },
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.85,
      "bounding_box": {
        "x1": 600,
        "y1": 80,
        "x2": 750,
        "y2": 380
      }
    }
  ]
}
```

## Backend Components

### 1. **detection_schema.py**
Defines all data structures for detection data:
- `BoundingBox`: Coordinates (x1, y1, x2, y2)
- `Detection`: Single object detection
- `DetectionFrame`: Collection of detections for one frame with serialization methods

### 2. **detection_channel.py**
`DetectionDataChannelManager` class:
```python
# Create and manage the data channel
manager = DetectionDataChannelManager(session_id)
manager.set_channel(rtc_channel)

# Send detections
await manager.send_detection_frame(detection_frame)

# Send status
manager.send_status_message("initializing", "Starting detection...")
```

### 3. **service.py Updates**
- Creates detection data channel when peer connection starts
- `_process_video_track()` now:
  1. Receives video frames
  2. Runs YOLO detection
  3. Converts results to `DetectionFrame`
  4. Sends via data channel
  5. Tracks frame_id for ordering

## Flow in Code

### Backend: Sending Detections

```python
# In _process_video_track() - service.py:

# 1. Get frame from WebRTC video track
frame = await track.recv()
img = frame.to_ndarray(format="bgr24")

# 2. Run YOLO detection on frame
detections = await asyncio.to_thread(detector.detect, img)

# 3. Create structured DetectionFrame
detection_frame = DetectionFrame.from_yolo_detections(
    frame_id=frame_id,
    timestamp=time.time(),
    yolo_detections=detections,  # List[YoloDetection]
)

# 4. Send to frontend via data channel
channel_manager = DETECTION_CHANNELS.get(session_id)
if channel_manager:
    await channel_manager.send_detection_frame(detection_frame)
```

The data is automatically serialized to JSON and sent through the WebRTC data channel labeled "detections".
```

## Configuration

In `.env` or environment variables:

```bash
# Enable YOLO detection
ENABLE_YOLO=True

# Detection frequency (frames per second)
DETECTION_FPS=5

# Model configuration
YOLO_MODEL_PATH=yolov8n.pt
YOLO_CONF_THRESHOLD=0.25
YOLO_MAX_WIDTH=640
```

## Error Handling

Backend logs for monitoring:

```
[DetectionChannel] Failed to send detection frame | session_id=... | error=...
[WebRTC] Video frame error: ...
[YOLO] detections=5 | session_id=... | person:0.92, car:0.88
[DetectionChannel] Channel opened | session_id=...
[DetectionChannel] Sent detections | session_id=... | objects=3
```

Common issues:
- **No detections sent**: Check if `ENABLE_YOLO=True` in environment
- **Channel not ready**: Frontend must establish connection first
- **Memory leaks**: Channels are cleaned up in `_cleanup()` method

## Performance Considerations

1. **Frame Rate**: Set `DETECTION_FPS` to balance accuracy vs CPU load
   - Lower = fewer inferences (faster)
   - Higher = more detections but higher CPU usage
   
2. **Model Size**: Trade-off between speed and accuracy
   ```bash
   YOLO_MODEL_PATH=yolov8n.pt   # Nano - fastest
   YOLO_MODEL_PATH=yolov8s.pt   # Small
   YOLO_MODEL_PATH=yolov8m.pt   # Medium
   ```

3. **Resolution**: `YOLO_MAX_WIDTH` downscales input for speed
   - Lower = faster inference, less accurate
   - Higher = slower, better accuracy

4. **Message Size**: Depends on detection count; typically 2-10KB per frame

## Backend Testing

### Manual Test of DetectionFrame
```python
# Test serialization locally
from app.modules.signaling.detection_schema import (
    DetectionFrame,
    Detection,
    BoundingBox,
)

# Create mock detections
detections = [
    Detection(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bounding_box=BoundingBox(100, 50, 300, 400),
    ),
    Detection(
        class_id=2,
        class_name="car",
        confidence=0.87,
        bounding_box=BoundingBox(400, 100, 700, 350),
    ),
]

# Create frame
frame = DetectionFrame(
    frame_id=1,
    timestamp=time.time(),
    detections=detections,
)

# Verify JSON serialization
json_str = frame.to_json()
print(json_str)

# Verify it's valid JSON
import json
data = json.loads(json_str)
print(f"Frame ID: {data['frame_id']}")
print(f"Objects: {data['detection_count']}")
```

## File Structure

```
backend/app/modules/signaling/
├── detection_schema.py          # Data structures
├── detection_channel.py         # Data channel manager
├── service.py                   # WebRTC service (updated)
├── interfaces.py
├── websocket_manager.py
├── schemas.py
└── DETECTION_FLOW.md           # This documentation
```

## Setup & Configuration

### Environment Variables

Add to `.env` or `example.env`:

```bash
# Detection settings
ENABLE_YOLO=True
DETECTION_FPS=5              # Process 5 frames per second
YOLO_MODEL_PATH=yolov8n.pt  # Model size
YOLO_CONF_THRESHOLD=0.25    # Confidence threshold
YOLO_MAX_WIDTH=640          # Max input width
```

### Dependencies

Ensure in `requirements.txt`:
```
ultralytics>=8.0.0    # YOLO
opencv-python>=4.5.0  # OpenCV
aiortc>=1.5.0         # WebRTC
```

## How It Works

1. **Connection**: Client establishes WebRTC peer connection
2. **Initialization**: Backend creates detection data channel automatically
3. **Processing**: Each video frame is:
   - Retrieved from WebRTC track
   - Downscaled if needed
   - Passed to YOLO detector
   - Results converted to `DetectionFrame`
4. **Transmission**: Structured JSON sent through data channel
5. **Frontend**: Receives and renders bounding boxes (in separate repo)

## Next Steps

1. Configure `.env` file with detection settings
2. Ensure `ENABLE_YOLO=True`
3. Install dependencies: `pip install -r requirements.txt`
4. Start backend - detections send automatically once peer connects
5. Check logs for `[DetectionChannel]` messages to confirm data flow
6. Implement frontend receiver (in separate frontend repo) to consume data

