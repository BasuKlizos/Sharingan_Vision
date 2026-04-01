# Frontend Detection Rendering Guide

## Overview
This guide explains how to properly render bounding boxes from YOLO detection data sent via WebRTC data channel to the frontend.

## Data Structure

The detection data arrives in this format via the `detections` data channel:

```json
{
  "type": "detection_frame",
  "frame_id": 123,
  "timestamp": 1711843200.123,
  "face": {
    "alerts": [],
    "faces": [],
    "face_count": 0
  },
  "yolo": {
    "detection_count": 5,
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bounding_box": {
          "x1": 100,
          "y1": 150,
          "x2": 450,
          "y2": 600
        }
      }
    ]
  }
}
```

## Coordinate System

- **Backend (YOLO)**: Returns bounding box coordinates in **original video frame dimensions** (e.g., 1280x720)
- **Frontend Canvas**: Your display canvas may have different dimensions
- **Key Issue**: Direct mapping of coordinates will cause misalignment if video dimensions ≠ canvas dimensions

## How to Properly Display Rectangles

### 1. Get Video Dimensions
When you receive the video stream, capture its original dimensions:

```javascript
const pc = new RTCPeerConnection();
let videoWidth = 0;
let videoHeight = 0;

pc.ontrack = (event) => {
  if (event.track.kind === 'video') {
    const video = document.getElementById('video');
    video.srcObject = event.streams[0];
    
    // Get dimensions when metadata loads
    video.onloadedmetadata = () => {
      videoWidth = video.videoWidth;
      videoHeight = video.videoHeight;
      console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);
    };
  }
};
```

### 2. Get Canvas Dimensions
```javascript
const canvas = document.getElementById('detection-canvas');
const ctx = canvas.getContext('2d');

const canvasWidth = canvas.width;
const canvasHeight = canvas.height;
```

### 3. Calculate Scale Factors
```javascript
function calculateScaleFactors(videoWidth, videoHeight, canvasWidth, canvasHeight) {
  const scaleX = canvasWidth / videoWidth;
  const scaleY = canvasHeight / videoHeight;
  
  return { scaleX, scaleY };
}

const { scaleX, scaleY } = calculateScaleFactors(videoWidth, videoHeight, canvasWidth, canvasHeight);
```

### 4. Draw Rectangles with Proper Scaling
```javascript
const dataChannel = pc.createDataChannel('detections');

dataChannel.onmessage = (event) => {
  try {
    const data = JSON.parse(event.data);
    
    if (data.type === 'detection_frame' && data.yolo) {
      // Clear previous detections
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      
      // Draw each detection
      data.yolo.detections.forEach(detection => {
        const { bounding_box, class_name, confidence } = detection;
        
        // Scale coordinates to canvas
        const x1 = bounding_box.x1 * scaleX;
        const y1 = bounding_box.y1 * scaleY;
        const x2 = bounding_box.x2 * scaleX;
        const y2 = bounding_box.y2 * scaleY;
        
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Draw rectangle
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw label
        const label = `${class_name} ${(confidence * 100).toFixed(1)}%`;
        const fontSize = 14;
        ctx.font = `${fontSize}px Arial`;
        ctx.fillStyle = '#00FF00';
        ctx.fillText(label, x1, y1 - 5);
      });
    }
  } catch (e) {
    console.error('Error processing detection:', e);
  }
};
```

## Alternative: Canvas with Video Overlay

If you're rendering the video directly on canvas (using `drawImage`), ensure alignment:

```javascript
// Setup video playback
const video = document.getElementById('video');
const canvas = document.getElementById('detection-canvas');
const ctx = canvas.getContext('2d');

// Set canvas size to match video display
function setupCanvas() {
  canvas.width = video.offsetWidth;
  canvas.height = video.offsetHeight;
  
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
}

// Draw video frame on canvas
function drawFrame() {
  // Draw video
  ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
  
  // Draw detections with scaling
  // ... (use the rectangle drawing code above)
  
  requestAnimationFrame(drawFrame);
}
```

## Common Issues & Solutions

### Issue 1: Rectangles too small or too large
**Cause**: Scale factors not calculated correctly
**Solution**: Verify `videoWidth`, `videoHeight`, `canvasWidth`, `canvasHeight` are all correct

```javascript
console.log(`Video: ${videoWidth}x${videoHeight}, Canvas: ${canvasWidth}x${canvasHeight}`);
console.log(`Scale factors: X=${scaleX}, Y=${scaleY}`);
```

### Issue 2: Rectangles offset to one side
**Cause**: Canvas has padding/margin or coordinate origin mismatch
**Solution**: 
- Check canvas CSS (should have `display: block` to avoid inline spacing)
- Verify coordinates start from (0,0)

```css
#detection-canvas {
  display: block;
  margin: 0;
  padding: 0;
}
```

### Issue 3: Rectangles not updating
**Cause**: Data channel not connected or message handler not working
**Solution**: 
- Verify `onmessage` is being triggered
- Check browser console for errors

```javascript
dataChannel.onopen = () => {
  console.log('Detection data channel opened');
};

dataChannel.onmessage = (event) => {
  console.log('Detection message received:', event.data);
  // ... process and draw
};
```

## Performance Tips

1. **Reuse canvas context** instead of creating new ones
2. **Clear canvas only the regions with detections** instead of full clear (if processing is slow)
3. **Debounce drawing** if receiving data faster than display refresh rate
4. **Draw on a separate canvas layer** to avoid redrawing video frame each time

```javascript
// Separate detection canvas overlay
const videoCanvas = document.getElementById('video-canvas');
const detectionCanvas = document.getElementById('detection-canvas');

// This way you only redraw detections, not the video
```

## Testing Checklist

- [ ] Video dimensions are correctly captured
- [ ] Canvas dimensions match display size
- [ ] Scale factors calculated and logged
- [ ] Rectangle coordinates are scaled before drawing
- [ ] Labels are positioned correctly
- [ ] Color contrast is visible
- [ ] Performance is acceptable (no lag)
- [ ] Works on different video resolutions
- [ ] Works on different screen sizes

## Example: Complete Implementation

```javascript
class DetectionRenderer {
  constructor(videoId, canvasId, dataChannelLabel = 'detections') {
    this.video = document.getElementById(videoId);
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.dataChannelLabel = dataChannelLabel;
    
    this.videoWidth = 0;
    this.videoHeight = 0;
    this.scaleX = 1;
    this.scaleY = 1;
  }

  setup(pc) {
    pc.ontrack = (event) => {
      if (event.track.kind === 'video') {
        this.video.srcObject = event.streams[0];
        this.video.onloadedmetadata = () => {
          this.videoWidth = this.video.videoWidth;
          this.videoHeight = this.video.videoHeight;
          this.resizeCanvas();
        };
      }
    };

    const dataChannel = pc.createDataChannel(this.dataChannelLabel);
    dataChannel.onmessage = (event) => this.handleDetection(event);
  }

  resizeCanvas() {
    this.canvas.width = this.video.offsetWidth;
    this.canvas.height = this.video.offsetHeight;
    
    this.scaleX = this.canvas.width / this.videoWidth;
    this.scaleY = this.canvas.height / this.videoHeight;
  }

  handleDetection(event) {
    try {
      const data = JSON.parse(event.data);
      this.render(data);
    } catch (e) {
      console.error('Detection error:', e);
    }
  }

  render(data) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    if (data.yolo?.detections) {
      data.yolo.detections.forEach((det) => {
        this.drawBox(det);
      });
    }
  }

  drawBox(detection) {
    const { bounding_box, class_name, confidence } = detection;
    
    const x1 = bounding_box.x1 * this.scaleX;
    const y1 = bounding_box.y1 * this.scaleY;
    const x2 = bounding_box.x2 * this.scaleX;
    const y2 = bounding_box.y2 * this.scaleY;
    
    const width = x2 - x1;
    const height = y2 - y1;
    
    this.ctx.strokeStyle = '#00FF00';
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(x1, y1, width, height);
    
    const label = `${class_name} ${(confidence * 100).toFixed(1)}%`;
    this.ctx.font = '14px Arial';
    this.ctx.fillStyle = '#00FF00';
    this.ctx.fillText(label, x1, y1 - 5);
  }
}

// Usage
const renderer = new DetectionRenderer('video', 'detection-canvas', 'detections');
renderer.setup(pc);
```

## Questions?

If rectangles still don't align properly:
1. Check the exact video resolution being sent
2. Verify canvas dimensions in browser DevTools
3. Log intermediate scale calculations
4. Compare bounding box coordinates with actual object positions
