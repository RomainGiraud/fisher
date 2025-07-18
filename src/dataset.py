import sys
import pathlib
import os
import shutil
import cv2
import pandas as pd
import numpy as np

raw_dir   = pathlib.Path(os.getcwd()) / "dataset" / "0 - raw"
crop_dir  = pathlib.Path(os.getcwd()) / "dataset" / "1 - cropped"
label_dir = pathlib.Path(os.getcwd()) / "dataset" / "2 - labelled"
final_dir = pathlib.Path(os.getcwd()) / "dataset" / "3 - final"

margin_top = 0.1
margin_bottom = 0.4
margin_left = 0.2
margin_right = 0.2

ignore_labels = ["1", "99"]

def crop_all():
  for file in raw_dir.glob("*.jpg"):
    img = cv2.imread(str(file))

    height, width = img.shape[:2]
    top = int(height * margin_top)
    bottom = int(height * (1 - margin_bottom))
    left = int(width * margin_left)
    right = int(width * (1 - margin_right))
    cropped_image = img[top:bottom, left:right]

    cv2.imwrite(str(crop_dir / file.name), cropped_image)

def split_all():
  train_dir = final_dir / "train"
  val_dir   = final_dir / "val"
  test_dir  = final_dir / "test"
  for subdir in [train_dir, val_dir, test_dir]:
    (subdir / "images").mkdir(parents=True, exist_ok=True)
    (subdir / "labels").mkdir(parents=True, exist_ok=True)

  images = []
  for file in (label_dir / "images").glob("*.jpg"):
    label = None
    with open(label_dir / "labels" / f"{file.stem}.txt", "r") as f:
      lines = f.readlines()
      if len(lines) == 0:
        label = "99"
      else:
        line = lines[0].strip().split(' ')
        label = line[0]
    if label in ignore_labels:
      continue
    images.append({"filename": file.name, "label": label})

  df = pd.DataFrame(images)
  grouped_df = df.groupby("label")
  arr_list = [np.split(g, [int(.8 * len(g)), int(.9 * len(g))]) for i, g in grouped_df]
  final_train = pd.concat([t[0] for t in arr_list])
  final_test = pd.concat([t[1] for t in arr_list])
  final_val = pd.concat([t[2] for t in arr_list])

  for name, dataset in {"train": final_train, "test": final_test, "val": final_val}.items():
    for idx, row in dataset.iterrows():
      shutil.copyfile(label_dir / "images" / row['filename'], final_dir / name / "images" / row['filename'])
      label_filename =  pathlib.Path(row['filename']).stem + ".txt"
      shutil.copyfile(label_dir / "labels" / label_filename, final_dir / name / "labels" / label_filename)

commands = {
  "crop": crop_all,
  "split": split_all,
}

if __name__ == "__main__":
  if len(sys.argv) < 2 or sys.argv[1] not in commands.keys():
    print("Usage: python dataset.py <command>")
    sys.exit(1)

  command = sys.argv[1]
  commands[command]()
