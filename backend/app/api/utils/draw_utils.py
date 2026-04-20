import cv2


def draw_landmarks(img, results):
    """
    Visualization layer → draws landmarks only
    """

    if not results.multi_face_landmarks:
        return img

    h, w, _ = img.shape

    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    return img
