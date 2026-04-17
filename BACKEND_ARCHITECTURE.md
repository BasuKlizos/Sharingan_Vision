# Backend Architecture - Sharingan Vision

## Executive Summary

This document describes the architecture of the Sharingan Vision backend, a FastAPI-based real-time proctoring and detection system. The backend processes video frames via WebRTC, performs object detection using YOLO, analyzes facial features using MediaPipe, and evaluates session integrity through a sophisticated rule-based proctoring engine.

**Key Characteristics:**
- Clean, layered architecture with clear separation of concerns
- Real-time WebRTC communication with frontend clients
- Async/await patterns for high-concurrency handling
- Modular detection and analysis pipelines
- Stateful session management with in-memory and persistent storage

---

## Table of Contents

1. [Architecture Principles](#architecture-principles)
2. [System Context & Components](#system-context--components)
3. [Clean Architecture Layers](#clean-architecture-layers)
4. [Module Hierarchy & Responsibilities](#module-hierarchy--responsibilities)
5. [Client-Backend Interaction](#client-backend-interaction)
6. [Real-Time Data Flow](#real-time-data-flow)
7. [Session Lifecycle](#session-lifecycle)
8. [State Management](#state-management)
9. [Error Handling & Resilience](#error-handling--resilience)
10. [Performance Considerations](#performance-considerations)

---

## Architecture Principles

### Clean Architecture Tenets

```
┌─────────────────────────────────────────────────────────┐
│                  CLEAN ARCHITECTURE LAYERS               │
│                                                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Presentation / API Layer (FastAPI Routes)         │ │
│  │  - HTTP endpoints                                  │ │
│  │  - Request/Response handling                       │ │
│  │  - Input validation                                │ │
│  └────────────────────────────────────────────────────┘ │
│                        ↑ ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Application / Use Case Layer (Services)           │ │
│  │  - Business logic orchestration                    │ │
│  │  - Workflow coordination                           │ │
│  │  - Dependency injection                            │ │
│  └────────────────────────────────────────────────────┘ │
│                        ↑ ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Domain / Entity Layer (Models, Schemas)           │ │
│  │  - Core business models                            │ │
│  │  - Value objects                                   │ │
│  │  - Domain logic                                    │ │
│  └────────────────────────────────────────────────────┘ │
│                        ↑ ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Infrastructure / Framework Layer                  │ │
│  │  - Database (MongoDB, Redis)                       │ │
│  │  - External services (WebRTC library)              │ │
│  │  - Configuration                                   │ │
│  └────────────────────────────────────────────────────┘ │
│                                                           │
└─────────────────────────────────────────────────────────┘

KEY PRINCIPLE: Outer layers depend on inner layers, 
never the reverse. Business logic is independent of 
frameworks and delivery mechanisms.
```

### Design Principles Applied

1. **Single Responsibility**: Each module has one reason to change
2. **Dependency Inversion**: High-level modules don't depend on low-level details
3. **Interface Segregation**: Small, focused interfaces
4. **Open/Closed**: Open for extension, closed for modification
5. **Async-First**: Leverage async/await for concurrent I/O

---

## System Context & Components

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SHARINGAN VISION SYSTEM                       │
│                                                                   │
│  ┌──────────────────┐                  ┌──────────────────────┐  │
│  │  FRONTEND CLIENT │                  │   BACKEND SERVER     │  │
│  │  (Next.js/React) │                  │    (FastAPI/Python)  │  │
│  │                  │                  │                      │  │
│  │ ┌──────────────┐ │                  │  ┌────────────────┐  │  │
│  │ │   Camera &   │ │                  │  │  API Router    │  │  │
│  │ │  MediaStream │ │                  │  │  /api/v1/...   │  │  │
│  │ └──────────────┘ │                  │  └────────────────┘  │  │
│  │        │         │                  │        │              │  │
│  │        ↓         │   HTTP (REST)    │        ↓              │  │
│  │ ┌──────────────┐ ├──────────────────┤  ┌────────────────┐  │  │
│  │ │  WebGazer &  │ │  Signaling       │  │  WebRTC        │  │  │
│  │ │  Calibration │ │  API Endpoints   │  │  Service       │  │  │
│  │ └──────────────┘ │                  │  │  (offerHandler)│  │  │
│  │        │         │                  │  └────────────────┘  │  │
│  │        ↓         │   HTTP (REST)    │        │              │  │
│  │ ┌──────────────┐ │                  │        ↓              │  │
│  │ │   Lighting   │ ├──────────────────┤  ┌────────────────┐  │  │
│  │ │   Precheck   │ │  Precheck API    │  │  Precheck      │  │  │
│  │ └──────────────┘ │  Endpoints       │  │  Service       │  │  │
│  │        │         │                  │  └────────────────┘  │  │
│  │        ↓         │   WebRTC         │        │              │  │
│  │ ┌──────────────┐ │   (Peer Conn +   │        ↓              │  │
│  │ │   Canvas     │ │   Data Channel)  │  ┌────────────────┐  │  │
│  │ │   Rendering  │ ├<────────────────→┤  │  Detection     │  │  │
│  │ │   Detection  │ │                  │  │  Pipeline      │  │  │
│  │ │   Display    │ │                  │  │                │  │  │
│  │ └──────────────┘ │                  │  ├────────────────┤  │  │
│  │                  │                  │  │  YOLO          │  │  │
│  │ ┌──────────────┐ │                  │  │  MediaPipe     │  │  │
│  │ │   Analytics  │ │                  │  │  Proctoring    │  │  │
│  │ │   Display    │ │                  │  └────────────────┘  │  │
│  │ └──────────────┘ │                  │        │              │  │
│  │                  │                  │        ↓              │  │
│  │                  │                  │  ┌────────────────┐  │  │
│  │                  │                  │  │  Storage       │  │  │
│  │                  │                  │  │  (Redis/Mongo) │  │  │
│  │                  │                  │  └────────────────┘  │  │
│  └──────────────────┘                  └──────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Clean Architecture Layers

### 1. Presentation Layer (API Endpoints)

**Location:** `app/api/`

**Responsibilities:**
- Define REST API endpoints
- Handle HTTP request/response serialization
- Validate incoming data
- Return properly formatted responses
- Handle HTTP error codes

**Key Files:**
- `app/api/router.py`: Main API router, includes sub-routers
- `app/api/v1/webrtc_signals.py`: WebRTC signaling endpoints
- `app/api/v1/precheck.py`: Precheck validation endpoints
- `app/api/utils/`: Utilities for route handling

**Example Endpoint Structure:**
```python
@router.post("/offer", response_model=AnswerResponse)
async def handle_offer(
    payload: OfferRequest,
    service: Annotated[WebRTCService, Depends(get_webrtc_service)],
):
    """
    HTTP Handler (Presentation Layer)
    ├─ Input validation (OfferRequest schema)
    ├─ Dependency injection (WebRTCService)
    ├─ Call business logic
    └─ Return formatted response (AnswerResponse)
    """
```

---

### 2. Application/Use Case Layer (Services)

**Location:** `app/modules/*/service.py`

**Responsibilities:**
- Orchestrate business workflows
- Coordinate between domain models and infrastructure
- Manage session state
- Implement use case logic
- Handle complex decision making

**Key Services:**

#### WebRTCService
- **Location:** `app/modules/signaling/service.py`
- **Responsibility:** Manages WebRTC sessions, peer connections, and frame processing
- **Key Methods:**
  - `handle_offer()`: Process SDP offer and return answer
  - `process_frame()`: Run detection pipeline on incoming frames
  - `send_detection_frame()`: Send results over data channel

#### LightingPrecheckService
- **Location:** `app/modules/precheck/service.py`
- **Responsibility:** Validate camera lighting conditions
- **Key Methods:**
  - `evaluate_frames()`: Analyze lighting quality
  - Thresholds: brightness range, dark/bright pixel ratios

#### ProctoringEngine
- **Location:** `app/modules/proctoring/engine.py`
- **Responsibility:** Apply proctoring rules and generate alerts
- **Key Methods:**
  - `update()`: Evaluate rules against current frame
  - Individual rule methods for different violation types

---

### 3. Domain Layer (Models, Schemas, Business Logic)

**Location:** `app/modules/*/schemas.py` and `app/modules/*/`

**Responsibilities:**
- Define core business entities
- Validate domain rules
- Express domain knowledge
- Support multiple representations (request/response)

**Key Domain Models:**

#### DetectionFrame
```python
@dataclass
class DetectionFrame:
    frame_id: int
    timestamp: float
    detections: List[Detection]
    crop_offset: CropOffset
    
    # Methods: to_dict(), to_json(), from_yolo_detections()
```

#### YoloDetection
```python
@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]  # Bounding box
```

#### ProctoringAlert
```python
@dataclass
class ProctoringAlert:
    session_id: str
    frame_id: int
    rule_id: str
    severity: str  # "low" | "medium" | "high" | "critical"
    message: str
    timestamp: float
```

#### SessionMonitoringRecord
```python
@dataclass
class SessionMonitoringRecord:
    session_id: str
    created_at: datetime
    webrtc_status: str
    calibration: Optional[CalibrationData]
    latest_current_view: Optional[CurrentViewData]
    latest_zone_assessment: Optional[dict]
```

---

### 4. Infrastructure Layer (Database, Config, External Services)

**Location:** `app/core/` and external libraries

**Responsibilities:**
- Manage database connections
- Handle external service communication
- Load configuration
- Provide system utilities

**Key Components:**

#### Redis Manager (`app/core/redis.py`)
- Singleton async Redis client
- Session-level caching
- Frame buffering
- Alert queue management

#### MongoDB Manager (`app/core/mongodb.py`)
- Async MongoDB connection
- Session persistence
- Alert logging
- Audit trail

#### Configuration (`app/core/config.py`)
- Environment variable loading
- Default settings
- Validation thresholds

---

## Module Hierarchy & Responsibilities

### Complete Module Structure

```
app/
├── api/                                 # Presentation Layer
│   ├── router.py                        # Main router
│   ├── v1/
│   │   ├── webrtc_signals.py            # POST /api/v1/webrtc/offer
│   │   └── precheck.py                  # POST /api/v1/precheck/lighting
│   ├── utils/
│   │   ├── webrtc_signals_utils.py      # Service getters
│   │   └── precheck_utils.py            # Service getters
│   └── monitoring.py                    # GET /api/monitoring/* (analytics)
│
├── modules/                             # Business Logic Layers
│   │
│   ├── signaling/                       # WebRTC & Detection
│   │   ├── service.py                   # WebRTCService (Use Case Layer)
│   │   ├── detection_channel.py         # Data channel management
│   │   ├── detection_schema.py          # DetectionFrame, BoundingBox (Domain)
│   │   ├── schemas.py                   # OfferRequest, AnswerResponse (API)
│   │   ├── DETECTION_FLOW.md            # Documentation
│   │   └── FRAME_PROCESSING_GUIDE.md    # Documentation
│   │
│   ├── detection/                       # Object Detection
│   │   ├── yolo_detector.py             # YOLO model inference
│   │   └── mediapipe_face.py            # Face detection & landmarks
│   │
│   ├── analytics/                       # Feature Extraction
│   │   ├── face_analyzer.py             # Face metrics, gaze, attention
│   │   └── head_movement.py             # Head pose analysis
│   │
│   ├── proctoring/                      # Integrity Monitoring
│   │   ├── engine.py                    # ProctoringEngine (rule evaluation)
│   │   ├── types.py                     # ProctoringAlert, ProctoringInputs
│   │   ├── store.py                     # Alert storage & retrieval
│   │   ├── flush_service.py             # Background alert flushing
│   │   └── *.py                         # Alert handlers, filters
│   │
│   ├── precheck/                        # Pre-Session Validation
│   │   ├── service.py                   # LightingPrecheckService
│   │   ├── schemas.py                   # Lighting validation schemas
│   │   └── PRECHECK_FRONTEND_GUIDE.md   # Documentation
│   │
│   ├── monitoring/                      # Session State & Analytics
│   │   ├── store.py                     # SessionMonitoringStore (in-memory)
│   │   ├── schemas.py                   # Calibration, CurrentView (Domain)
│   │   ├── zones.py                     # Zone assessment logic
│   │   └── __init__.py                  # Store initialization
│   │
│   ├── decision/                        # (Future) Decision Logic
│   └── interview/                       # (Future) Interview Management
│
├── core/                                # Infrastructure Layer
│   ├── config.py                        # Settings & environment
│   ├── redis.py                         # Redis client manager
│   ├── mongodb.py                       # MongoDB client manager
│   └── singleton.py                     # Singleton metaclass
│
├── dependencies/                        # Dependency Injection
│   ├── redis.py                         # Redis dependency
│   └── mongodb.py                       # MongoDB dependency
│
├── middleware/                          # Request/Response Middleware
│   └── log_middleware.py                # Request logging
│
├── common/                              # Shared Utilities
│   └── exceptions.py                    # Custom exception types
│
├── cron/                                # Background Tasks
│   └── health_check.py                  # Periodic health checks
│
├── main.py                              # FastAPI app initialization
├── logger.py                            # Logging configuration
└── __init__.py
```

### Module Responsibilities Matrix

| Module | Responsibility | Layer | Input | Output |
|--------|-----------------|-------|-------|--------|
| `webrtc_signals` | Handle SDP offer/answer | API | OfferRequest | AnswerResponse + session_id |
| `WebRTCService` | Manage peer connections & frame processing | Application | SDPs, video frames | Answers, detection frames |
| `detection_schema` | Structure detection data | Domain | Raw detections | DetectionFrame objects |
| `YoloDetector` | Run YOLO inference | Domain/Infra | Video frame (numpy) | List[YoloDetection] |
| `MediaPipeFaceDetector` | Extract face landmarks | Domain/Infra | Video frame (numpy) | Face landmarks, contours |
| `FaceAnalyzer` | Analyze gaze, attention, movement | Domain | Face landmarks | Face metrics, alerts |
| `ProctoringEngine` | Evaluate integrity rules | Application | Frame data + analysis | List[ProctoringAlert] |
| `LightingPrecheckService` | Validate lighting conditions | Application | Base64 frames | Lighting assessment |
| `SessionMonitoringStore` | Track session state | Infrastructure | Session events | Session records |
| `DetectionDataChannelManager` | Send detection over WebRTC | Infrastructure | DetectionFrame | Data channel message |

---

## Client-Backend Interaction

### Complete Request-Response Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                   CLIENT-BACKEND INTERACTION FLOW                 │
└──────────────────────────────────────────────────────────────────┘

PHASE 1: WEBRTC SIGNALING
─────────────────────────────

Client                                    Backend
  │                                         │
  │  POST /api/v1/webrtc/offer              │
  │  Body: { sdp, type: "offer" }           │
  ├────────────────────────────────────────→│
  │                                    [Application Layer]
  │                                    WebRTCService.handle_offer()
  │                                         │
  │                                    1. Create RTCPeerConnection
  │                                    2. Create detection data channel
  │                                    3. Set remote description (offer)
  │                                    4. Create SDP answer
  │                                    5. Generate session_id
  │                                         │
  │  Response: { sdp, type, session_id }   │
  │←────────────────────────────────────────┤
  │                                         │
  │  RTCPeerConnection established ←─────→ ICE negotiation
  │  Remote video track arrives             │
  │                                         │

PHASE 2: LIGHTING PRECHECK (Optional)
─────────────────────────────────────

Client                                    Backend
  │                                         │
  │  POST /api/v1/precheck/lighting         │
  │  Body: { frames: [base64...] }         │
  ├────────────────────────────────────────→│
  │                                    [Application Layer]
  │                                    LightingPrecheckService
  │                                         │
  │                                    1. Decode base64 frames
  │                                    2. Convert to OpenCV format
  │                                    3. Measure brightness/contrast
  │                                    4. Compare against thresholds
  │                                         │
  │  Response: { ok, status, level }       │
  │←────────────────────────────────────────┤
  │                                         │

PHASE 3: ACTIVE SESSION (CONTINUOUS)
───────────────────────────────────────

Client                                    Backend
  │                                         │
  │  Video Track (RTC)  ────────────────→  │
  │  (Camera stream)                        │
  │                                         │
  │                              [Frame Processing Pipeline]
  │                                         │
  │                                    For each frame:
  │                                    1. Receive video frame
  │                                    2. YOLO detection
  │                                    3. MediaPipe face detection
  │                                    4. FaceAnalyzer analysis
  │                                    5. ProctoringEngine evaluation
  │                                    6. Build DetectionFrame
  │                                    7. Send via data channel
  │                                         │
  │  Detection Frame (Data Channel) ←────  │
  │  { detections, face_data, alerts }     │
  │                                         │
  │  Current View Update (Data Channel)     │
  │  { type: "current_view", ... }  ────→ │
  │                                    [Session Monitoring]
  │                                    SessionMonitoringStore
  │                                    .update_current_view()
  │                                         │
  │  ↑ Loop repeats at video frame rate    │

PHASE 4: SESSION TEARDOWN
──────────────────────────

Client                                    Backend
  │                                         │
  │  RTCPeerConnection.close()  ────────→  │
  │  Stop video/audio tracks                │
  │                                    [Cleanup]
  │                                    1. Close data channel
  │                                    2. Remove peer connection
  │                                    3. Save session summary
  │                                    4. Cleanup resources
  │                                         │
  │  Confirmation (or automatic)            │
  │←────────────────────────────────────────┤
```

### API Endpoint Details

#### 1. WebRTC Offer Endpoint

**Path:** `POST /api/v1/webrtc/offer`

**Request:**
```json
{
  "sdp": "v=0\r\no=- ... (full SDP offer)",
  "type": "offer"
}
```

**Response (200 OK):**
```json
{
  "sdp": "v=0\r\no=- ... (full SDP answer)",
  "type": "answer",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Processing Flow:**
```
request: OfferRequest
    ↓
[Presentation Layer]
validate_request()
    ↓
[Application Layer]
WebRTCService.handle_offer()
  ├── Create RTCPeerConnection
  ├── Setup event handlers
  ├── Create data channel
  ├── Add local audio/video tracks
  ├── Set remote description (offer)
  ├── Create SDP answer
  ├── Set local description
  ├── Generate session_id
  └── Store connection in PEER_CONNECTIONS dict
    ↓
response: AnswerResponse
```

**Error Handling:**
- `400 Bad Request`: Invalid SDP format or schema validation failure
- `500 Internal Server Error`: Exception during offer processing

---

#### 2. Lighting Precheck Endpoint

**Path:** `POST /api/v1/precheck/lighting`

**Request:**
```json
{
  "frames": [
    {
      "base64_data": "iVBORw0KGgo...",
      "timestamp": 1711427195.3842
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "ok": true,
  "status": "ok",
  "brightness_level": 0.65,
  "condition": "well_lit",
  "frames": [
    {
      "index": 0,
      "valid": true,
      "status": "ok",
      "brightness_mean": 130.5,
      "dark_pixel_ratio": 0.15,
      "bright_pixel_ratio": 0.10
    }
  ]
}
```

**Processing Flow:**
```
request: LightingPrecheckRequest
    ↓
[Application Layer]
LightingPrecheckService.evaluate_frames()
  ├── For each base64 frame:
  │   ├── Decode base64
  │   ├── Convert to OpenCV image
  │   ├── Calculate brightness histogram
  │   ├── Count dark/bright pixels
  │   └── Compare against thresholds
  ├── Aggregate results
  ├── Decide overall status (ok/too_dark/too_bright)
  └── Return LightingPrecheckResponse
    ↓
response: LightingPrecheckResponse
```

---

#### 3. Monitoring/Analytics Endpoints

**Paths:**
- `GET /api/monitoring/sessions/{session_id}`: Get session metrics
- `GET /api/monitoring/sessions/{session_id}/alerts`: Get session alerts
- `GET /api/health`: Health check

**Response Example:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-04-17T10:30:00Z",
  "duration_seconds": 1200,
  "frames_processed": 24000,
  "alerts_triggered": 15,
  "detection_stats": {
    "people_count": 1,
    "avg_confidence": 0.92,
    "objects_detected": ["person", "laptop"]
  }
}
```

---

## Real-Time Data Flow

### Frame Processing Pipeline

```
┌────────────────────────────────────────────────────────┐
│              FRAME PROCESSING PIPELINE                  │
└────────────────────────────────────────────────────────┘

   INCOMING VIDEO FRAME (from RTCPeerConnection)
   ├─ Source: Client camera stream
   ├─ Format: YUV420, H264 encoded, RTC track
   └─ Frequency: 24-30 fps
        ↓
   [1] FRAME EXTRACTION
   ├─ Extract from RTC track
   ├─ Decode to OpenCV RGB image
   └─ Shape: (H, W, 3), dtype=uint8
        ↓
   [2] CROP & SCALE (Optional)
   ├─ Apply center crop if enabled
   ├─ Resize to model input size
   └─ Record crop offset metadata
        ↓
   [3] YOLO DETECTION
   ├─ Input: RGB image
   ├─ Model: YOLOv8n/m/l/x variants
   ├─ Output: List[YoloDetection]
   │  └─ class_id, class_name, confidence, xyxy
   └─ Processing time: 20-100ms depending on model
        ↓
   [4] MEDIAPIPE FACE DETECTION
   ├─ Input: RGB image
   ├─ Output: Face landmarks (468 points)
   │  ├─ Eyes, nose, mouth, face contour
   │  └─ Normalized coordinates (0-1)
   └─ Processing time: 10-30ms
        ↓
   [5] FACE ANALYSIS
   ├─ Input: Face landmarks
   ├─ Analyzer: FaceAnalyzer instance
   ├─ Compute:
   │  ├─ Eye direction (left/center/right)
   │  ├─ Head pose (yaw, pitch, roll)
   │  ├─ Head movement speed
   │  ├─ Gaze stability
   │  └─ Blink detection
   └─ Output: face_analysis dict
        ↓
   [6] PROCTORING ENGINE
   ├─ Input: YOLODetections + face_analysis
   ├─ Engine: ProctoringEngine.update()
   ├─ Evaluate rules:
   │  ├─ THIRD_PARTY_PRESENCE (multiple faces)
   │  ├─ PHONE_DETECTED (cell phone in frame)
   │  ├─ EXTERNAL_DEVICE_DETECTED (laptop, tablet)
   │  ├─ GAZE_AWAY (looking at screen edge)
   │  ├─ EXCESSIVE_HEAD_MOVEMENT (sudden movement)
   │  ├─ SUSPICIOUS_HAND_MOVEMENT (hands near face)
   │  ├─ ABNORMAL_BLINK (too frequent/slow)
   │  └─ REPEATED_EYE_CLOSURE (closed eyes)
   └─ Output: List[ProctoringAlert]
        ↓
   [7] DETECTION FRAME ASSEMBLY
   ├─ Create DetectionFrame object
   ├─ Fields:
   │  ├─ frame_id: Sequential counter
   │  ├─ timestamp: Unix timestamp
   │  ├─ detections: List[Detection] (YOLO results)
   │  ├─ crop_offset: CropOffset metadata
   │  └─ face_analysis: Optional face data
   └─ Serialize to JSON
        ↓
   [8] DATA CHANNEL TRANSMISSION
   ├─ Send DetectionFrame as JSON string
   ├─ Over: RTCDataChannel (ordered, unreliable)
   ├─ Payload size: 5-50 KB per frame
   └─ Rate: 24-30 fps (one per input frame)
        ↓
   [9] ALERT HANDLING
   ├─ If ProctoringAlert generated:
   │  ├─ Store in ProctoringStore
   │  ├─ Enqueue for flush_service
   │  ├─ Optionally persist to MongoDB
   │  └─ Emit via monitoring APIs
   └─ If no alerts: Silent pass
        ↓
   [10] SESSION MONITORING
   ├─ Update SessionMonitoringStore
   ├─ Track:
   │  ├─ Frame count
   │  ├─ Average detection confidence
   │  ├─ Objects detected in session
   │  └─ Alert timeline
   └─ Store for analytics query
        ↓
   LOOP: Repeat for next frame

   Total Pipeline Latency: 50-200ms (model dependent)
   Bottleneck: YOLO inference (20-100ms)
```

### Concurrent Processing

**Challenge:** Single request handler → multiple frames → blocking operations

**Solution:** Async/await with concurrent processing

```python
# WebRTCService processes frames concurrently
async def process_video_frame(frame_data):
    # All I/O is non-blocking
    
    # Parallel operations where possible
    yolo_task = asyncio.create_task(run_yolo(frame))
    face_task = asyncio.create_task(run_mediapipe(frame))
    
    yolo_results, face_results = await asyncio.gather(yolo_task, face_task)
    
    # Sequential dependent operations
    face_analysis = await analyze_face(face_results)
    alerts = await proctor_engine.update(yolo_results, face_analysis)
    
    # Send detections (non-blocking)
    await send_detection_frame(yolo_results, face_analysis, alerts)
```

**Benefits:**
- Non-blocking I/O allows many concurrent clients
- Efficient CPU usage through async event loop
- Can handle 10+ simultaneous WebRTC sessions

---

## Session Lifecycle

### Complete Session State Machine

```
┌─────────────────────────────────────────────────────────┐
│              SESSION STATE MACHINE                       │
└─────────────────────────────────────────────────────────┘

                  ┌──────────────┐
                  │   CREATED    │
                  │ (session_id  │
                  │  generated)  │
                  └──────┬───────┘
                         │
         [Client sends SDP offer]
                         │
                         ↓
                  ┌──────────────┐
                  │   OFFERED    │
                  │ (SDP offer   │
                  │  received)   │
                  └──────┬───────┘
                         │
        [Backend creates RTCPeerConnection]
        [Backend creates SDP answer]
                         │
                         ↓
                  ┌──────────────┐
                  │  ANSWERED    │
                  │ (SDP answer  │
                  │  sent)       │
                  └──────┬───────┘
                         │
      [ICE negotiation in progress]
                         │
                         ↓
                  ┌──────────────┐
                  │ CONNECTING   │
                  │ (ICE checks  │
                  │  running)    │
                  └──────┬───────┘
                         │
       [ICE candidates succeed]
       [Peer connection established]
                         │
                         ↓
                  ┌──────────────┐
                  │ CONNECTED    │
                  │ (video flow) │
                  │ (detections) │
                  │ (active)     │
                  └──────┬───────┘
                         │
    [Video frames arriving and being processed]
    [Detection frames being sent]
    [Alerts being generated if conditions met]
                         │
                  [Session continues...]
                         │
         [Client closes connection OR timeout]
                         │
                         ↓
                  ┌──────────────┐
                  │ DISCONNECTED │
                  │ (connection  │
                  │  closed by   │
                  │  peer)       │
                  └──────┬───────┘
                         │
         [Backend cleanup scheduled]
         [Resources released]
         [Session summary saved]
                         │
                         ↓
                  ┌──────────────┐
                  │    CLOSED    │
                  │ (session     │
                  │  archived)   │
                  └──────────────┘
```

### Session Data Storage

**In-Memory (Global Dicts):**
```python
# Global state trackers in WebRTCService
PEER_CONNECTIONS: Dict[str, RTCPeerConnection] = {}
DETECTION_CHANNELS: Dict[str, DetectionDataChannelManager] = {}
```

**Session Monitoring Store (In-Memory):**
```python
# Thread-safe session tracking
SessionMonitoringStore:
  _sessions: Dict[session_id, SessionMonitoringRecord]
  _lock: RLock  # For thread safety
```

**Persistent Storage (Optional):**
```python
# MongoDB for long-term archival
mongo_manager.collection("sessions").insert_one({
  "session_id": session_id,
  "created_at": ISO8601,
  "duration": seconds,
  "frame_count": count,
  "alerts": List[alert],
  "summary": {...}
})

# Redis for caching frequently accessed data
redis_client.setex(
  f"session:{session_id}:data",
  3600,  # 1 hour TTL
  json_encoded_session_data
)
```

---

## State Management

### Global Application State

```
┌────────────────────────────────────────────────────────┐
│             GLOBAL APPLICATION STATE                    │
└────────────────────────────────────────────────────────┘

FastAPI App Scope:
├── Single instance per process
├── Initialized once at startup (lifespan context)
└── Shared across all requests

  ├─ PEER_CONNECTIONS (dict)
  │  ├─ Key: session_id
  │  ├─ Value: RTCPeerConnection object
  │  ├─ Lifecycle: Created on offer, destroyed on disconnect
  │  └─ Thread-safe: Protected by acquisition of RTC lock
  │
  ├─ DETECTION_CHANNELS (dict)
  │  ├─ Key: session_id
  │  ├─ Value: DetectionDataChannelManager object
  │  ├─ Lifecycle: Created with peer connection
  │  └─ Manages: Data channel for detection frames
  │
  ├─ session_monitoring_store (SessionMonitoringStore)
  │  ├─ Singleton instance
  │  ├─ Thread-safe: Uses RLock
  │  └─ Tracks: Calibration, current_view, alerts per session
  │
  ├─ WebRTCService.face_analyzers (dict)
  │  ├─ Key: session_id
  │  ├─ Value: FaceAnalyzer instance
  │  └─ Lifecycle: One per session, persists for duration
  │
  ├─ WebRTCService.proctor_engines (dict)
  │  ├─ Key: session_id
  │  ├─ Value: ProctoringEngine instance
  │  └─ Tracks: Rule states, hold times, cooldowns
  │
  └─ WebRTCService.data_channels (dict)
     ├─ Key: session_id
     ├─ Value: RTCDataChannel reference
     └─ Used for: Sending detection frames

Connection Scope:
├── Per WebRTC connection (per client)
├── Includes: Local state, frame buffers, temporary data
└── Cleanup: On connection close
```

### State Transitions & Safety

**Thread Safety:**
- Python's GIL protects basic operations
- Session monitoring uses RLock for explicit synchronization
- Dictionary operations are atomic for simple get/set

**Cleanup Strategy:**
```python
# Scheduled cleanup to prevent resource leaks
async def cleanup_session(session_id: str):
    # Mark for cleanup
    _CLEANUP_SCHEDULED.add(session_id)
    
    try:
        # Close connections
        if session_id in PEER_CONNECTIONS:
            await PEER_CONNECTIONS[session_id].close()
            del PEER_CONNECTIONS[session_id]
        
        # Close data channels
        if session_id in DETECTION_CHANNELS:
            await DETECTION_CHANNELS[session_id].cleanup()
            del DETECTION_CHANNELS[session_id]
        
        # Save final session state
        record = session_monitoring_store.get_session(session_id)
        if record:
            await persist_to_mongodb(record)
    
    finally:
        _CLEANUP_SCHEDULED.discard(session_id)
```

**Memory Management:**
- Frame buffers limited to recent frames only
- Old frames automatically discarded
- Long-running sessions: Periodic cleanup every 5 minutes
- Alert history: Configurable retention (default: 10,000 most recent)

---

## Error Handling & Resilience

### Error Categories

#### 1. WebRTC Errors

```
┌─ ICE Failure
│  ├─ Symptom: ICE connection state → "failed"
│  ├─ Cause: Network unreachable, firewall blocked
│  ├─ Recovery: Inform client, suggest firewall/network check
│  └─ Handling: Close connection, cleanup, return 500
│
├─ SDP Parse Error
│  ├─ Symptom: Invalid SDP string
│  ├─ Cause: Malformed offer from client
│  ├─ Recovery: Return 400 Bad Request with error details
│  └─ Log: Warning level, include received SDP length
│
└─ Data Channel Error
   ├─ Symptom: Channel.onerror fired
   ├─ Cause: Peer closed unexpectedly, transport failure
   ├─ Recovery: Attempt to continue, log issue
   └─ Handling: Keep connection alive, skip data transmission
```

#### 2. Detection Pipeline Errors

```
┌─ YOLO Inference Failure
│  ├─ Symptom: Model throws exception
│  ├─ Cause: Corrupted frame, OOM, device failure
│  ├─ Recovery: Log error, skip frame, continue
│  └─ Impact: One frame of missing detections
│
├─ MediaPipe Failure
│  ├─ Symptom: Face detection returns None
│  ├─ Cause: No face in frame (valid), process issue (rare)
│  ├─ Recovery: Continue with no-face state
│  └─ Impact: Some face-based rules disabled for frame
│
└─ Frame Processing Timeout
   ├─ Symptom: Frame processing exceeds threshold (e.g., >500ms)
   ├─ Cause: System overload, GPU contention
   ├─ Recovery: Log warning, skip frame, continue
   └─ Impact: Frame rate reduction
```

#### 3. Storage Errors

```
┌─ Redis Connection Loss
│  ├─ Recovery: Graceful degradation, use in-memory store
│  └─ Impact: Alert persistence only to local buffer
│
└─ MongoDB Write Failure
   ├─ Recovery: Retry with exponential backoff
   └─ Impact: Session history not saved (non-critical)
```

### Error Handling Patterns

**Pattern 1: Graceful Degradation**
```python
async def send_detection_frame(session_id, frame):
    try:
        channel = DETECTION_CHANNELS[session_id]
        if channel.is_ready:
            await channel.send(frame.to_json())
    except Exception as e:
        logger.error(f"Failed to send detection: {e}")
        # Continue processing, don't crash
```

**Pattern 2: Retry with Backoff**
```python
async def persist_alert(alert):
    for attempt in range(3):
        try:
            await mongo_manager.alerts.insert_one(alert.to_dict())
            return
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
            else:
                logger.error(f"Alert persistence failed after retries: {e}")
```

**Pattern 3: Fallback**
```python
def get_session_data(session_id):
    # Try Redis cache first
    try:
        data = redis_client.get(f"session:{session_id}")
        if data:
            return json.loads(data)
    except:
        pass
    
    # Fallback to in-memory store
    record = session_monitoring_store.get_session(session_id)
    return record.to_dict() if record else None
```

---

## Performance Considerations

### Performance Targets

```
Metric                  | Target        | Acceptable Range
------------------------+---------------+------------------
WebRTC Setup Time       | <2 seconds    | 1-3 seconds
Frame Processing Latency| <100ms        | 50-200ms
Detection FPS           | 24-30         | 15-30
Data Channel Throughput | 10-50 Mbps    | Depends on network
Memory Per Session      | <200 MB       | <500 MB
Concurrent Sessions     | 10+           | Hardware dependent
Alert Latency           | <1 second     | <2 seconds
```

### Bottleneck Analysis

**CPU-Bound:**
```
YOLO Inference: 20-100ms per frame (dominant)
├─ Model size matters: n < m < l < x
├─ Input resolution affects time quadratically
└─ Solution: Use lighter models, reduce resolution

MediaPipe: 10-30ms per frame
├─ Usually fast on CPU
└─ GPU acceleration available

FaceAnalyzer: <5ms per frame
├─ Pure math operations
└─ Negligible impact
```

**I/O-Bound:**
```
WebRTC Data Channel Send: 1-10ms
├─ Mostly waiting for buffer availability
└─ Non-blocking, queued automatically

Redis/MongoDB Writes: 10-100ms
├─ Network + disk latency
├─ Async operations don't block inference
└─ Background flush_service handles queueing
```

**Memory:**
```
Per Session Memory Usage:
├─ RTCPeerConnection: ~50-100 MB
├─ Video buffers: ~30-50 MB
├─ DetectionFrameBuffer (30 frames): ~5-10 MB
├─ Face landmarks buffer: <1 MB
└─ Total per session: ~150-200 MB

Optimization:
├─ Limit frame buffer size (max 30 frames)
├─ Discard frames older than 5 seconds
├─ Use object pooling for frame objects
└─ Periodically GC old frames
```

### Scalability Considerations

**Vertical Scaling (Single Server):**
- Max sessions: Limited by GPU VRAM (8GB GPU → ~5-10 sessions)
- CPU bottleneck: 10+ cores needed for 10 sessions
- Network: 100 Mbps upload per session @ 30fps, 1080p

**Horizontal Scaling (Multiple Servers):**
```
Architecture:
┌──────────────────────────────────────────────────────┐
│                   Load Balancer                       │
│              (Round-robin, sticky)                    │
└────────────────────────────────────────────────────────┘
    │           │           │           │
    ↓           ↓           ↓           ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Server 1│ │Server 2│ │Server 3│ │Server N│
│ (5 s)  │ │ (5 s)  │ │ (5 s)  │ │ (5 s)  │
└────────┘ └────────┘ └────────┘ └────────┘
    │           │           │           │
    └───────────┴───────────┴───────────┘
           ↓
    ┌──────────────────┐
    │  Shared Redis    │  (Alert queue, session cache)
    │  Shared MongoDB  │  (Persistent storage)
    └──────────────────┘
```

**Constraints:**
- Session affinity required (sticky sessions)
- Redis/MongoDB become bottleneck at scale
- WebRTC doesn't scale across servers (per-server only)

---

## Summary

### Architecture Highlights

✅ **Clean Architecture:**
- Clear separation: API → Services → Domain → Infrastructure
- Business logic independent of frameworks
- Testable, maintainable, extensible

✅ **Real-Time Processing:**
- Async/await throughout
- Concurrent client handling
- Sub-100ms latency target

✅ **Reliability:**
- Graceful error handling
- Automatic cleanup and resource management
- Persistent storage with fallbacks

✅ **Observability:**
- Comprehensive logging at each layer
- Session monitoring and analytics
- Alert tracking and history

✅ **Modularity:**
- Independent detector modules (YOLO, MediaPipe)
- Pluggable analysis components (FaceAnalyzer)
- Rule-based proctoring engine (extensible)

### Future Enhancement Opportunities

1. **Horizontal Scaling:** Session migration between servers
2. **GPU Optimization:** Batch processing across sessions
3. **Model Variants:** Dynamic model selection based on load
4. **Advanced Analytics:** ML-based anomaly detection
5. **Multi-Modal Proctoring:** Audio/behavioral analysis
6. **Real-Time Dashboard:** Live monitoring of all sessions
7. **Custom Rule Engine:** User-defined proctoring rules

---

## Appendix: Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | FastAPI | 0.100+ |
| WebRTC | aiortc | 1.5+ |
| Detection | YOLOv8 | ultralytics |
| Face Analysis | MediaPipe | 0.10+ |
| Async | asyncio | Python 3.10+ |
| Database | MongoDB | 4.4+ |
| Cache | Redis | 6.0+ |
| Logging | Python logging | stdlib |
| Config | Pydantic | 2.0+ |

---

## Appendix: Configuration Parameters

Key settings in `app/core/config.py`:

```python
# WebRTC
YOLO_MODEL_PATH = "yolov8n.pt"  # Model choice
YOLO_CONF_THRESHOLD = 0.25      # Detection threshold

# Precheck
PRECHECK_MIN_BRIGHTNESS = 70    # Brightness bounds
PRECHECK_MAX_BRIGHTNESS = 190
PRECHECK_MAX_DARK_RATIO = 0.35  # Dark pixel ratio

# Proctoring
PROCTOR_MULTI_FACE_HOLD_SECONDS = 0.5  # Hold window
PROCTOR_ALERT_COOLDOWN_SECONDS = 5     # Alert throttle

# Storage
PROCTOR_STORE_BACKEND = "dual"  # "mongo", "redis", "dual"
ALERT_RETENTION_SIZE = 10000    # Max alerts per session

# Ports
REDIS_HOST = "localhost"
REDIS_PORT = 6379
MONGODB_URL = "mongodb://localhost:27017"
```

This architecture provides a solid foundation for building a reliable, scalable, and maintainable real-time proctoring system.
