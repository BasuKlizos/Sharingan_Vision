# Lighting Precheck Frontend Guide

## Purpose

Run a lighting precheck before starting WebRTC.

This must be a separate HTTP flow.

Do not:

- use WebRTC for precheck
- use a data channel for precheck
- mix precheck with the real-time detection loop

## Required Frontend Flow

1. Start camera preview.
2. Capture 3 to 5 frames from the preview.
3. Send those frames to `POST /api/v1/precheck/lighting`.
4. Read the JSON response.
5. If `ok === true`, start WebRTC.
6. If `ok === false`, show the returned message and let the user retry.

## API Endpoint

```text
POST /api/v1/precheck/lighting
Content-Type: application/json
```

Example full URL:

```text
http://localhost:8000/api/v1/precheck/lighting
```

## Request Body

Send JSON with a `frames` array.

Each item should be a base64 image string.

Accepted formats:

- plain base64
- data URL like `data:image/jpeg;base64,...`

Example:

```json
{
  "frames": [
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,..."
  ]
}
```

## Response Body

The API returns JSON like this:

```json
{
  "ok": true,
  "status": "ok",
  "message": "Lighting looks good. You can start WebRTC.",
  "checked_frames": 3,
  "valid_frames": 3,
  "summary": {
    "brightness_mean": 121.42,
    "dark_pixel_ratio": 0.0842,
    "bright_pixel_ratio": 0.0112,
    "min_brightness": 70.0,
    "max_brightness": 190.0,
    "max_dark_ratio": 0.35,
    "max_bright_ratio": 0.25
  },
  "frames": [
    {
      "index": 0,
      "valid": true,
      "status": "ok",
      "message": "Lighting looks good. You can start WebRTC.",
      "brightness_mean": 120.17,
      "dark_pixel_ratio": 0.0911,
      "bright_pixel_ratio": 0.0104
    }
  ]
}
```

## Frontend Decision Rules

- If `status === "ok"`, continue to WebRTC startup.
- If `status === "too_dark"`, block WebRTC and ask the user to increase lighting.
- If `status === "too_bright"`, block WebRTC and ask the user to reduce glare or backlight.
- If `status === "invalid_frames"`, block WebRTC and retry frame capture.

## Frontend Notes

- Use the existing camera preview to capture frames.
- Prefer JPEG to keep the payload smaller.
- Keep the preview visible while precheck is running.
- Show `message` from the API directly in the UI.
- Keep this logic in a separate precheck function or hook.
- Only start WebRTC after precheck succeeds.

## Final Rule

The sequence must stay:

```text
Camera preview
-> capture frames
-> call /precheck/lighting
-> check JSON result
-> if ok, start WebRTC
```
