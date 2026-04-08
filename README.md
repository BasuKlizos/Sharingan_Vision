## Sharingan Vision

### Proctoring (Redis-backed)
During WebRTC video processing, the backend evaluates rule-based cheating indicators and persists:
- Minimal per-frame metrics (no raw landmarks / bounding boxes)
- Emitted alerts (rule id, severity, risk score, evidence)

**Redis keys**
- `proctor:metrics:{session_id}` (capped list)
- `proctor:alerts:{session_id}` (capped list)

**MongoDB (optional)**
- Set `MONGODB_URI` (default `mongodb://localhost:27017`) and `MONGODB_DB`
- Set `PROCTOR_STORE_BACKEND=mongo` to persist proctoring metrics/alerts in MongoDB collections:
  - `proctor_metrics`
  - `proctor_alerts`

**API (mounted under `/api`)**
- `GET /api/proctoring/{session_id}/alerts?limit=50`
- `GET /api/proctoring/{session_id}/metrics?limit=200`
