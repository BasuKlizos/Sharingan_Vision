from app.logger import logger

class FaceAnalyzer:
    """
    Analysis layer → converts landmarks into behavior insights
    """

    def __init__(self):
        # State for smoothing & stability
        self.prev_states = {}
        self.no_face_counter = 0

    def analyze(self, results, img_shape, session_id=None):
        alerts = []
        faces_data = []

        # 1. NO FACE HANDLING (BUFFERED)
        if not results.multi_face_landmarks:
            self.no_face_counter += 1

            # Avoid flicker for 1–2 missed frames
            if self.no_face_counter < 3:
                return {
                    "alerts": [],
                    "faces": [],
                    "face_count": 0
                }

            logger.debug(f"[Analyzer] No face detected | session_id={session_id}")
            return {
                "alerts": ["NO_FACE"],
                "faces": [],
                "face_count": 0
            }

        # Reset counter when face detected
        self.no_face_counter = 0

        face_count = len(results.multi_face_landmarks)

        logger.debug(
            f"[Analyzer] Faces detected: {face_count} | session_id={session_id}"
        )

        # 2. MULTIPLE FACES CHECK
        if 1 < face_count <= 3:
            alerts.append("MULTIPLE_FACES")

        h, w, _ = img_shape

        # 3. PROCESS EACH FACE
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):

            # Use EYES instead of nose (more stable)
            LEFT_EYE = 33
            RIGHT_EYE = 263

            left_eye = face_landmarks.landmark[LEFT_EYE]
            right_eye = face_landmarks.landmark[RIGHT_EYE]

            # Center of eyes
            eye_center_x = (left_eye.x + right_eye.x) / 2
            eye_center_y = (left_eye.y + right_eye.y) / 2

            # Convert to pixel
            eye_x_px = int(eye_center_x * w)
            eye_y_px = int(eye_center_y * h)

            # 4. LOOKING AWAY LOGIC (IMPROVED
            center_offset = abs(eye_center_x - 0.5)

            # threshold tuned for stability
            looking_away_raw = center_offset > 0.12

            # 5. SMOOTHING (ANTI-FLICKER)
            prev_state = self.prev_states.get(idx, False)

            # simple debounce
            if looking_away_raw != prev_state:
                looking_away = prev_state
            else:
                looking_away = looking_away_raw

            self.prev_states[idx] = looking_away

            if looking_away:
                alerts.append("LOOKING_AWAY")

            logger.debug(
                f"[Analyzer] Face {idx+1} | looking_away={looking_away} "
                f"| offset={center_offset:.2f} "
                f"| eye_px=({eye_x_px},{eye_y_px}) "
                f"| session_id={session_id}"
            )

            faces_data.append({
                "looking_away": looking_away,
                "eye_norm": (float(eye_center_x), float(eye_center_y)),
                "eye_px": (eye_x_px, eye_y_px)
            })

        # 6. FINAL RESPONSE
        return {
            "alerts": list(set(alerts)),
            "faces": faces_data,
            "face_count": face_count
        }