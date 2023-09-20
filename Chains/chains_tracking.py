
import os
from typing import List
import shutil

import cv2
import numpy as np
import matplotlib.image as mpim

from chains_detector import ChainDetector
from tracker import Tracker
from data import Result
import data as dat
import preprocessing

def max_intensity_video(image_list: List[str]) -> int:
    """Detect the maximum intensity in a video."""
    max_int = 0
    for im_name in image_list:
        im = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
        max_int = max(max_int, np.amax(im))
    return max_int

def convert_16to8bits(image: str, i: int, max_int: int) -> None:
    """Convert 16bit image to 8bit and store it in a temp folder."""
    splitted = image.split("/")
    im_name = splitted[-1]
    new_name = image.replace(im_name, f"tmp/Image{i:07d}.png")
    im16 = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if im16.dtype == "uint16":
        im8 = (im16 * 0.98 * 2 ** 8 / max_int).astype("uint8")
    else:
        im8 = im16
    cv2.imwrite(new_name, im8)

folder_path = "/Users/sintes/Desktop/Martyna/PhD/chaines/2020-12-08_13h01m43s"
tmp = os.path.join(folder_path, "tmp")
try:
    os.mkdir(tmp)
except FileExistsError:
    pass

image_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".tif")]
max_int = max_intensity_video(image_list)
for i, im in enumerate(image_list):
    convert_16to8bits(im, i, max_int)

image_list = [os.path.join(tmp, file) for file in os.listdir(tmp) if file.endswith(".png")]

processed_path = os.path.join(folder_path, "Figure", "Processed")
tracked_path = os.path.join(folder_path, "Figure","Tracked")
shutil.rmtree(processed_path, ignore_errors=True)
shutil.rmtree(tracked_path, ignore_errors=True)
os.makedirs(tracked_path)
os.makedirs(processed_path)

# Load configuration
config = dat.Configuration()
params = config.read_toml("/Users/sintes/Documents/Python/Chains/Chains/cfg.toml")
# Data saver
saver = Result(folder_path)

# Set up detector
detector = ChainDetector(params, processed_path, visualisation=True)
background = preprocessing.get_background(image_list)
cv2.imwrite(os.path.join(folder_path, "Figure/background.png"), background)
detector.set_background(background)


# Set up tracker
tracker = Tracker(params, tracked_path) 
tracker.set_params(params)
tracker.set_detector(detector)

camera = cv2.VideoCapture(
    "{}/Image%07d.png".format(tmp))
# camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
# camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
im_data = tracker.initialize(cv2.cvtColor(camera.read()[1], cv2.COLOR_BGR2GRAY))
saver.add_data(im_data)

ret = True
while (ret):
    ret, frame = camera.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_data = tracker.process(frame)
        saver.add_data(im_data)

camera.release()
shutil.rmtree(tmp)