import pyautogui
import pyscreeze
import cv2
import numpy as np
from PIL import ImageGrab

import os
import pathlib
from abc import ABC, abstractmethod

import utils

def main():
    # pyautogui.alert(text='Hello, World!', title='Greeting', button='OK')
    screenWidth, screenHeight = pyautogui.size()
    print(f"Screen size: {screenWidth}x{screenHeight}")

    margin_x = int(screenWidth * 0.3)
    margin_top    = int(screenHeight * 0.1)
    margin_bottom = int(screenHeight * 0.5)
    region=(margin_x, margin_top, screenWidth - (margin_x * 2), screenHeight - margin_top - margin_bottom)
    pyautogui.screenshot('my_screenshot.png', region=region)
    pyautogui.sleep(5)

    # while True:
    #     mouse = pyautogui.position()
    #     print(f"Current mouse position: {mouse}")
    #     if mouse.x > region[0] and mouse.x < region[0] or mouse.y
    #     pyautogui.sleep(1)  # Wait for 1 second before checking again

    while True:
        pyautogui.press('9')
        pyautogui.sleep(1)

        # position = None
        positions = []
        while True:
            try:
                # position = pyautogui.locateCenterOnScreen('ref_02.png', confidence=0.7, region=region, grayscale=True)
                found = pyautogui.locateAllOnScreen('ref_01.png', confidence=0.6, region=region, grayscale=False)
                for position in found:
                    positions.append(pyautogui.center(position))
                print("Image found on the screen!")
                break # TODO
            except pyautogui.ImageNotFoundException:
                print("Image not found, checking again...")
            except pyscreeze.ImageNotFoundException:
                print("Image not found, checking again...")
            pyautogui.sleep(1)  # Wait for 1 second before checking again

        print(f"Current mouse position: {pyautogui.position()}")
        # if position:
        for position in positions:
            print(f"Image position: {position}")
            pyautogui.moveTo(position.x, position.y)
            # print(f"Current mouse position: {pyautogui.position()}, current found image position: {position}")
            # pyautogui.rightClick()
            pyautogui.sleep(1)
        pyautogui.sleep(3)

class Detector(ABC):
    def __init__(self):
        self.agents = []
        self.preprocessors = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_preprocessor(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def run(self):
        print("Starting screen detection...")
        # Here you would implement the logic to start the detection process
        while True:
            image = self.get_image()
            if image is None:
                print("No image captured, retrying...")
                pyautogui.sleep(1)
                continue

            for preprocessor in self.preprocessors:
                image = preprocessor.apply(image)

            positions = []
            for agent in self.agents:
                positions.extend(agent.detect(image))

            positions = np.array(positions, dtype=np.int32).reshape(-1, 4)
            # print(f"Detected positions: {len(positions)}")
            positions = utils.non_max_suppression(positions, overlapThresh=0.1)
            # print(f"Cleaning positions: {len(positions)}")

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
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return frame

    def found(self, positions):
        if not positions:
            return False
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
        img = self.image.copy()
        h, w = img.shape[:2]
        for pt in positions:
            cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (0,0,255), 1)
        cv2.imwrite(f'output/ImageDetector-{self.idx}.png', img)
        self.idx += 1

class FishingBot:
    def __init__(self):
        pass

from agents.MatchBest import MatchBest
from agents.MatchAll import MatchAll
from preprocessors.Region import Region

if __name__ == "__main__":
    dir = pathlib.Path(os.path.dirname(__file__))
    detector = ImageDetector(dir / '../test_images/Screenshot 2025-07-06 214359.png')
    detector.add_preprocessor(Region(margin_top=0.1, margin_bottom=0.5, margin_left=0.3, margin_right=0.3))
    detector.add_agent(MatchAll(reference_image=cv2.imread(dir / '../reference_images/ref_04.png', cv2.IMREAD_COLOR_BGR)))
    detector.run()

    # bot = FishingBot(detector)
    # bot.start()
