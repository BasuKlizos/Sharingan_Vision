import cv2
import mediapipe as mp

from app.logger import logger

class MediaPipeFaceDetector:
    def __init__(self, max_faces: int = 5):
        """
        Initialize MediaPipe FaceMesh detector
        """
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True
        )

        logger.info(f"[MediaPipe] Initialized FaceMesh | max_faces={max_faces}")

    def process(self, img, session_id: str = None):
        """
        Process frame and return:
        - processed image
        - metadata (faces, alerts)
        """

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        alerts = []
        faces_data = []

        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)

            logger.debug(
                f"[MediaPipe] Faces detected: {face_count} | session_id={session_id}"
            )

            if face_count > 1:
                alerts.append("MULTIPLE_FACES")

            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                h, w, _ = img.shape

                # Draw landmarks
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

                nose = face_landmarks.landmark[1]
                looking_away = nose.x < 0.3 or nose.x > 0.7

                if looking_away:
                    alerts.append("LOOKING_AWAY")

                logger.debug(
                    f"[MediaPipe] Face {idx+1} | looking_away={looking_away} "
                    f"| nose_x={nose.x:.2f} | session_id={session_id}"
                )

                faces_data.append({
                    "looking_away": looking_away,
                    "nose_x": float(nose.x)
                })

        else:
            alerts.append("NO_FACE")

            logger.debug(
                f"[MediaPipe] No face detected | session_id={session_id}"
            )

        # Remove duplicate alerts
        alerts = list(set(alerts))

        return img, {
            "alerts": alerts,
            "faces": faces_data,
            "face_count": len(faces_data)
        }
