import cv2
import numpy as np

class MatchAll:
    def __init__(self, reference_image, use_grayscale=False, threshold=0.7):
        self.reference_image = reference_image
        self.use_grayscale = use_grayscale
        self.threshold = threshold

    def detect(self, image):
        if self.use_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(image, self.reference_image, cv2.TM_CCOEFF_NORMED)

        result2 = np.where(result >= self.threshold)
        result2 = np.sort(result2)
        h, w = self.reference_image.shape[:2]

        positions = []
        for (x, y) in zip(result2[1], result2[0]):
            positions.append((x, y, x + w, y + h))

        return positions
