
import os
import shutil

import cv2

from chains_detector import ChainDetector
from tracker import Tracker
from data import Result
import data as dat
import preprocessing


folder_path = "/Users/sintes/Desktop/ImageSeq"
image_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".tif")]

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
detector.set_background(background)

# Set up tracker
tracker = Tracker(params)
tracker.set_params(params)
tracker.set_detector(detector)

camera = cv2.VideoCapture(
    "{}/Chains%04d.tif".format(folder_path))
dat = tracker.initialize(cv2.cvtColor(camera.read()[1], cv2.COLOR_BGR2GRAY))
saver.add_data(dat)

ret = True
while (ret):
    ret, frame = camera.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dat = tracker.process(frame)
        saver.add_data(dat)

camera.release()
