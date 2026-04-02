from typing import Any

import mediapipe as mp
import numpy as np


class MediaPipeFaceDetector:
    """
    Detection layer that runs MediaPipe FaceMesh and returns raw results.
    """

    def __init__(self, max_faces: int = 1):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
        )

    def detect(self, img: np.ndarray) -> Any:
        """
        Run MediaPipe FaceMesh on an RGB image.

        Args:
            img: RGB image as numpy array with shape (H, W, 3)

        Returns:
            Raw MediaPipe FaceMesh result object.
        """
        return self.face_mesh.process(img)