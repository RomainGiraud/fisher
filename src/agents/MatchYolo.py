import cv2
import numpy as np
import ultralytics
import os

class MatchYolo:
    def __init__(self):
        self.model = ultralytics.YOLO('best.pt')

    def detect(self, image):
        results = self.model([image])
        for result in results:
            print(result)
            if result.boxes:
                print(f"Detected {len(result.boxes)} objects.")
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    print(f"Box coordinates: ({x1}, {y1}), ({x2}, {y2})")
                    # You can return or process the bounding boxes as needed
                    return [(x1, y1, x2, y2)]
            else:
                print("No objects detected.")
        return []
