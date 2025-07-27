# Fisher

## Description

Fisher is an automated fishing bot for World of Warcraft, designed to detect the fishing bobber on the screen and interact with the game by simulating mouse and keyboard actions. The project leverages computer vision (OpenCV, PIL), automation (pyautogui), and custom pattern detection agents.

## Features

- Screen capture and image processing using OpenCV and PIL
- Modular agent system for different detection strategies: pattern matching (OpenCV) and ML detection (Yolo)
- Automated mouse and keyboard actions with pyautogui
- Non-maximum suppression for filtering overlapping detections
- Configurable detection region
- Hotkey support for pausing or cancelling the bot

## Requirements

Install dependencies:

```bash
uv sync
```

## Usage

1. Launch the game and position your character for fishing: near a lake, in first-person view (FPS), with a clear visibility of the water.
2. Run the bot:

```bash
uv run
```

3. The bot will start after a short delay, cast the fishing line, and attempt to detect the bobber. When the bobber moves, the bot will right-click to catch the fish.
4. Press the configured hotkey (default: `esc`) to stop the bot.

## Train a model on your own data

1. Populate `dataset/0 - raw` with in-game screenshots (at least 100 images in different environments).
2. Run preprocessor: `uv run dataset.py crop`, this script will crop images to `dataset/1 - cropped` folder.
3. Run label-studio: `uv run label-studio`, create a new project, import all cropped images and label them.
4. Export label-studio project to "Yolo with images" format to `dataset/2 - labelled` folder.
5. Run split script: `uv run dataset.py split`, datasets will be created in `dataset/3 - final` folder.

Now you can train a model:
```bash
uv run yolo train model=yolo11n.pt data=dataset/3\ -\ final/data.yaml epochs=100
```

You just have to move the new model from `runs` folder to `models/best-fisher.pt`.

To anonymize your data:
```bash
for i in *.jpg;
do
  magick $i -fill black -draw 'rectangle 10,10 %[fx:w*0.15],%[fx:h*0.1]' output.jpg;
  mv output.jpg $i;
done
```

## Disclaimer

This project is for educational purposes only. Automating gameplay violates the terms of service. Use at your own risk.
