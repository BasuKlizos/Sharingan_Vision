# Frontend Crop Offset Guide - Detection Rendering

## Overview

The backend processes video frames with optional cropping for performance optimization. When a frame is cropped:

1. **Original frame dimensions**: Full video stream (e.g., 1280×720)
2. **Cropped frame**: Center portion (e.g., 1024×576) for faster processing
3. **Detections**: Run on cropped frame, coordinates are relative to cropped region
4. **Solution**: Apply crop offset to transform coordinates back to original frame positions

## Data Format

The backend sends detection data in this structure:

```json
{
  "type": "detection_frame",
  "frame_id": 42,
  "timestamp": 1711872345.123,
  
  "crop_offset": {
    "x_offset": 128,           // Left crop pixels
    "y_offset": 72,            // Top crop pixels
    "original_width": 1280,    // Full video width
    "original_height": 720,    // Full video height
    "cropped_width": 1024,     // Cropped region width
    "cropped_height": 576      // Cropped region height
  },
  
  "yolo": {
    "detection_count": 2,
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.92,
        "bounding_box": {
          "x1": 150,  // Left (relative to CROPPED image)
          "y1": 100,  // Top
          "x2": 350,  // Right
          "y2": 400   // Bottom
        }
      }
    ]
  },
  
  "face": {
    "face_count": 1,
    "faces": [{...}],
    "alerts": []
  }
}
```

## Key Insight: Coordinate Transformation

**Raw detection coordinates** are relative to the **cropped image**. You must add the crop offset to get positions in the **original video frame**:

```
Original Position = Detection Position + Crop Offset

original_x = detection_x + crop_offset.x_offset
original_y = detection_y + crop_offset.y_offset
```

## Implementation: Canvas Rendering with Crop Offsets

### Basic Example

```javascript
// Receive detection data
function handleDetectionData(data) {
  const { yolo, crop_offset } = data;
  
  if (!yolo.detections.length) return;
  
  // Get canvas and context
  const canvas = document.getElementById('detectionCanvas');
  const ctx = canvas.getContext('2d');
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Process each detection
  for (const detection of yolo.detections) {
    const bbox = detection.bounding_box;
    
    // ✅ CORRECT: Apply crop offset to get original coordinates
    const x1_original = bbox.x1 + crop_offset.x_offset;
    const y1_original = bbox.y1 + crop_offset.y_offset;
    const x2_original = bbox.x2 + crop_offset.x_offset;
    const y2_original = bbox.y2 + crop_offset.y_offset;
    
    // Draw rectangle
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    ctx.strokeRect(
      x1_original, 
      y1_original, 
      x2_original - x1_original, 
      y2_original - y1_original
    );
    
    // Draw label
    ctx.fillStyle = '#00FF00';
    ctx.font = 'bold 14px Arial';
    ctx.fillText(
      `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`,
      x1_original + 5,
      y1_original - 5
    );
  }
}
```

### With Canvas Scaling (Recommended for responsive UI)

If your canvas size differs from video dimensions, you need both offset AND scaling:

```javascript
class DetectionRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.videoWidth = 1280;
    this.videoHeight = 720;
  }
  
  render(detectionData) {
    const { yolo, crop_offset } = detectionData;
    
    // Calculate scale factors (canvas size vs video size)
    const scaleX = this.canvas.width / this.videoWidth;
    const scaleY = this.canvas.height / this.videoHeight;
    
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw each detection
    for (const detection of yolo.detections) {
      const bbox = detection.bounding_box;
      
      // Apply crop offset
      const x1_original = bbox.x1 + crop_offset.x_offset;
      const y1_original = bbox.y1 + crop_offset.y_offset;
      const x2_original = bbox.x2 + crop_offset.x_offset;
      const y2_original = bbox.y2 + crop_offset.y_offset;
      
      // Apply canvas scaling
      const x1_canvas = x1_original * scaleX;
      const y1_canvas = y1_original * scaleY;
      const x2_canvas = x2_original * scaleX;
      const y2_canvas = y2_original * scaleY;
      
      const width = x2_canvas - x1_canvas;
      const height = y2_canvas - y1_canvas;
      
      // Draw rectangle
      this.drawBox(x1_canvas, y1_canvas, width, height, detection);
    }
  }
  
  drawBox(x, y, width, height, detection) {
    // Rectangle
    this.ctx.strokeStyle = '#00FF00';
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(x, y, width, height);
    
    // Background for text
    this.ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
    const textHeight = 20;
    this.ctx.fillRect(x, y - textHeight, 200, textHeight);
    
    // Label
    this.ctx.fillStyle = '#000000';
    this.ctx.font = 'bold 14px Arial';
    const confidence = (detection.confidence * 100).toFixed(0);
    this.ctx.fillText(`${detection.class_name} ${confidence}%`, x + 5, y - 5);
  }
}

// Usage
const renderer = new DetectionRenderer('detectionCanvas');

dataChannel.onmessage = (event) => {
  const detectionData = JSON.parse(event.data);
  renderer.render(detectionData);
};
```

## Common Issues & Debugging

### Issue 1: Boxes appear in wrong positions

**Cause**: Not applying crop offset

**Fix**: Always add `crop_offset.x_offset` and `crop_offset.y_offset` to detection coordinates

```javascript
// ❌ WRONG
ctx.strokeRect(bbox.x1, bbox.y1, ...);

// ✅ CORRECT
ctx.strokeRect(
  bbox.x1 + crop_offset.x_offset, 
  bbox.y1 + crop_offset.y_offset, 
  ...
);
```

### Issue 2: Boxes appear cut off or partially visible

**Cause**: Canvas size doesn't match video dimensions + missing scaling

**Fix**: Apply both offset AND scaling:

```javascript
const x_original = bbox.x1 + crop_offset.x_offset;
const x_canvas = x_original * (canvas.width / crop_offset.original_width);
```

### Issue 3: Boxes don't appear at all

**Cause 1**: `crop_offset` is null or undefined
```javascript
// Safe handling
const offset_x = crop_offset?.x_offset || 0;
const offset_y = crop_offset?.y_offset || 0;
```

**Cause 2**: Drawing outside canvas bounds
```javascript
// Clamp to canvas boundaries
x1_canvas = Math.max(0, Math.min(x1_canvas, canvas.width));
y1_canvas = Math.max(0, Math.min(y1_canvas, canvas.height));
```

## Backend Configuration

To enable/disable cropping, set these environment variables:

```bash
# .env
ENABLE_CROP=true              # Enable cropping optimization
CROP_PERCENT=0.8              # Keep 80% center (crop 20% total)
ENABLE_YOLO=true              # Enable YOLO detection
DETECTION_FPS=10              # Process 10 frames/second
```

When `ENABLE_CROP=false`:
- `crop_offset.x_offset = 0`
- `crop_offset.y_offset = 0`
- `crop_offset.original_width = cropped_width`
- `crop_offset.original_height = cropped_height`

(Your rendering code works with or without cropping - always apply the offset!)

## Formula Reference

```
// Transform from cropped coordinates to original video frame
x_original = x_cropped + crop_offset.x_offset
y_original = y_cropped + crop_offset.y_offset

// Scale to custom canvas size
scale_x = canvas.width / crop_offset.original_width
scale_y = canvas.height / crop_offset.original_height

x_canvas = x_original * scale_x
y_canvas = y_original * scale_y
```

## Testing Checklist

- [ ] Load a video stream and receive detection data
- [ ] Check that `crop_offset` values are included in payload
- [ ] Draw boxes on canvas
- [ ] Verify boxes align with actual objects in video
- [ ] Test with cropping enabled (`ENABLE_CROP=true`)
- [ ] Test with cropping disabled (`ENABLE_CROP=false`)
- [ ] Test with different canvas sizes (responsive resize)
- [ ] Test with multiple detections
- [ ] Verify labels display correctly with confidence scores

## See Also

- [Backend Crop Offset Documentation](./backend/app/modules/signaling/DETECTION_FLOW.md)
- [Detection Schema](./backend/app/modules/signaling/detection_schema.py)
