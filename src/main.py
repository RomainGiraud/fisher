import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import keyboard

import os
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime

import utils

class Detector(ABC):
    def __init__(self):
        self.agents = []
        self.image = None

    def add_agent(self, agent):
        self.agents.append(agent)

    def set_margins(self, margin_top=0, margin_bottom=0, margin_left=0, margin_right=0):
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.margin_left = margin_left
        self.margin_right = margin_right

    def step(self):
        image = self.get_image()
        if image is None:
            print("No image captured, retrying...")
            return []

        height, width = image.shape[:2]
        top = int(height * self.margin_top)
        bottom = int(height * (1 - self.margin_bottom))
        left = int(width * self.margin_left)
        right = int(width * (1 - self.margin_right))
        image = image[top:bottom, left:right]

        positions = []
        for agent in self.agents:
            positions.extend(agent.detect(image))

        for idx, position in enumerate(positions):
            positions[idx] = (
                position[0] + left,
                position[1] + top,
                position[2] + left,
                position[3] + top,
            )

        positions = np.array(positions, dtype=np.int32).reshape(-1, 4)
        # print(f"Detected positions: {len(positions)}")
        positions = utils.non_max_suppression(positions, overlapThresh=0.1)
        # print(f"Cleaning positions: {len(positions)}")
        return positions

    def run(self):
        print("Starting screen detection...")
        # Here you would implement the logic to start the detection process
        while True:
            positions = self.step()
            self.found(positions)
            pyautogui.sleep(1)

    @abstractmethod
    def get_image(self):
        pass

    def found(self, positions):
        print(f"Detected positions: {len(positions)}")

class ScreenDetector(Detector):
    def __init__(self):
        super().__init__()

    def get_image(self):
        # Grab the screen (PIL Image)
        img = ImageGrab.grab()
        # Convert to numpy array (RGB)
        img_np = np.array(img)
        # Convert RGB to BGR for OpenCV9
        self.image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return self.image

    def found(self, positions):
        # Here you can implement logic to handle found positions
        print(f"Found {len(positions)} positions on the screen.")
        for pos in positions:
            print(f"Position: {pos}")
        return True

class ImageDetector(Detector):
    def __init__(self, image_path=None):
        super().__init__()
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.idx = 0

    def get_image(self):
        return self.image

    def found(self, positions):
        utils.save_detection(f'output/ImageDetector-{self.idx:0>3}.png', self.image, positions)
        self.idx += 1

class FishingBot:
    def __init__(self, detector: Detector):
        self.detector = detector
        self.is_cancelled = False
        keyboard.on_press_key('esc', lambda _: setattr(self, 'is_cancelled', True))

    def start(self):
        idx = 0
        pyautogui.sleep(3)  # Wait for 3 seconds before starting
        prev_position = None
        while not self.is_cancelled:
            pyautogui.press('9')
            pyautogui.sleep(2)

            time = datetime.now()
            while True:
                pyautogui.sleep(0.1)

                if (datetime.now() - time).total_seconds() > 30:
                    break

                positions = detector.step()
                print(f"Detected positions ({len(positions)}): {positions}")
                idx += 1
                utils.save_detection(f'output/detect-{idx:0>3}.png', self.detector.image, positions)

                if len(positions) == 0:
                    if prev_position is None:
                        continue
                    else:
                        # float disappears
                        break

                pos = utils.box_center(positions[0])
                print(f"Selected position: {positions[0]} / {pos}")
                utils.save_detection(f'output/selected-{idx:0>3}.png', self.detector.image, [positions[0]])
                if prev_position is None:
                    prev_position = pos
                    continue

                if abs(pos[1] - prev_position[1]) > 10 or abs(pos[0] - prev_position[0]) > 10:
                    # new position found, float has moved
                    break

            if prev_position is not None:
                bbox = [prev_position[0] - 10, prev_position[1] - 10, prev_position[0] + 10, prev_position[1] + 10]
                utils.save_detection(f'output/click-{idx:0>3}.png', self.detector.image, [bbox])

                print(f"Clicking at position: {prev_position}")
                pyautogui.moveTo(prev_position[0], prev_position[1])
                pyautogui.rightClick()

if __name__ == "__main__":
    from agents.MatchBest import MatchBest
    from agents.MatchAll import MatchAll

    dir = pathlib.Path(os.path.dirname(__file__))
    # detector = ImageDetector(dir / '../test_images/Screenshot 2025-07-06 214359.png')
    detector = ScreenDetector()
    detector.set_margins(margin_top=0.1, margin_bottom=0.5, margin_left=0.3, margin_right=0.3)
    detector.add_agent(MatchAll(reference_image=cv2.imread(dir / '../reference_images/ref_04.png', cv2.IMREAD_COLOR_BGR)))
    # detector.run()

    bot = FishingBot(detector)
    bot.start()
