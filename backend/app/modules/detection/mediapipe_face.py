from typing import Any

import cv2
import mediapipe as mp

class MediaPipeFaceDetector:
    """
    Detection layer → ONLY runs MediaPipe and returns raw results
    """


    def __init__(self, max_faces: int = 2):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True
        )

    def detect(self, img: Any) -> Any:
        results = self.face_mesh.process(img)
        return results

