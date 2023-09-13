import cv2
import os

from chains_detector import ChainDetector
from tracker import Tracker
from data import Result
import data as dat
import preprocessing


folder_path = "/Users/sintes/Desktop/ImageSeq"
image_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".tif")]

# Load configuration
config = dat.Configuration()
params = config.read_toml("/Users/sintes/Documents/Python/Chains/Chains/cfg.toml")
# Data saver
saver = Result("{}".format(folder_path))

# Set up detector
detector = ChainDetector(params)
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
