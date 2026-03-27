from app.logger import logger

class FaceAnalyzer:
    """
    Analysis layer → converts landmarks into behavior insights
    """

    def analyze(self, results, img_shape, session_id=None):
        alerts = []
        faces_data = []

        if not results.multi_face_landmarks:
            logger.debug(f"[Analyzer] No face detected | session_id={session_id}")
            return {
                "alerts": ["NO_FACE"],
                "faces": [],
                "face_count": 0
            }

        face_count = len(results.multi_face_landmarks)

        logger.debug(
            f"[Analyzer] Faces detected: {face_count} | session_id={session_id}"
        )

        if face_count > 1:
            alerts.append("MULTIPLE_FACES")

        h, w, _ = img_shape

        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            nose = face_landmarks.landmark[1]

            # Convert to pixel coordinates
            nose_x_px = int(nose.x * w)
            nose_y_px = int(nose.y * h)

            # Use pixel position for logic (more stable)
            looking_away = nose.x < 0.3 or nose.x > 0.7

            if looking_away:
                alerts.append("LOOKING_AWAY")

            logger.debug(
                f"[Analyzer] Face {idx+1} | looking_away={looking_away} "
                f"| nose_norm=({nose.x:.2f},{nose.y:.2f}) "
                f"| nose_px=({nose_x_px},{nose_y_px}) "
                f"| session_id={session_id}"
            )

            faces_data.append({
                "looking_away": looking_away,
                "nose_norm": (float(nose.x), float(nose.y)),
                "nose_px": (nose_x_px, nose_y_px)
            })

        return {
            "alerts": list(set(alerts)),
            "faces": faces_data,
            "face_count": face_count
        }
