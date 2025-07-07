import cv2
import numpy as np

class MatchBest:
    def __init__(self, reference_image, use_grayscale=False, threshold=None):
        self.reference_image = reference_image
        self.use_grayscale = use_grayscale
        self.threshold = threshold

    def detect(self, image):
        if self.use_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
        #         'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
        # for meth in methods:
        #     method = getattr(cv2, meth)

        # Perform template matching
        result = cv2.matchTemplate(image, self.reference_image, cv2.TM_CCOEFF_NORMED)

        result2 = np.reshape(result, result.shape[0] * result.shape[1])
        sort = np.argsort(result2)
        (y1, x1) = np.unravel_index(sort[0], result.shape) # best match

        if self.threshold is not None and result[y1, x1] < self.threshold:
            return []

        h, w = self.reference_image.shape[:2]
        return [(x1, y1, x1 + w, y1 + h)]
