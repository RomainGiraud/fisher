import cv2
import numpy as np
import ultralytics
import os

class MatchYolo:
    def __init__(self, model_path):
        self.model = ultralytics.YOLO(model_path)

    def detect(self, image):
        positions = []
        results = self.model.predict(source=image, conf=0.5, save=False, verbose=False)
        result = results[0]
        if result.boxes:
            # print(f"Detected {len(result.boxes)} objects.")
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                positions.append((x1, y1, x2, y2))
        else:
            # print("No objects detected.")
            pass
        return positions
