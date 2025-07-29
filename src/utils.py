# import the necessary packages
import numpy as np
import os
import importlib
import cv2

def non_max_suppression(boxes, overlapThresh=0.3):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  # if the bounding boxes are integers, convert them to floats -- this
  # is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  # initialize the list of picked indexes
  pick = []
  count = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]

  # compute the area of the bounding boxes and grab the indexes to sort
  # (in the case that no probabilities are provided, simply sort on the
  # bottom-left y-coordinate)
  area = (x2 - x1 + 1) * (y2 - y1 + 1)

  # sort the indexes
  idxs = np.argsort(y2)

  # keep looping while some indexes still remain in the indexes list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the index value
    # to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of the bounding
    # box and the smallest (x, y) coordinates for the end of the bounding
    # box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have overlap greater
    # than the provided overlap threshold
    to_delete = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
    idxs = np.delete(idxs, to_delete)
    count.append(len(to_delete))

  # return only the bounding boxes that were picked
  # ordered by the most overlapped boxes
  pick = np.array(pick)
  count = np.array(count)
  indices = count.argsort()
  count = count[indices[::-1]]
  pick = pick[indices[::-1]]
  return boxes[pick].astype("int")

def load_class_from_file(file_path, class_name):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def load_agent(name, kwargs={}):
    agent_path = f'src/agents/{name}.py'
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent file '{agent_path}' does not exist.")

    try:
        AgentClass = load_class_from_file(agent_path, name)
        return AgentClass(**kwargs)
    except Exception as e:
        print(f"Error loading agent '{name}': {e}")
        return None

def save_detection(filename, image, positions):
    img = image.copy()
    h, w = img.shape[:2]
    for pt in positions:
        cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (0,0,255), 1)
    cv2.imwrite(filename, img)

def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2)), (abs(x2 - x1), abs(y2 - y1))
