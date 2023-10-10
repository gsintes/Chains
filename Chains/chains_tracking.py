
import os
from typing import List, Tuple
import shutil
import multiprocessing as mp

import cv2
import numpy as np

from chains_detector import ChainDetector
from tracker import Tracker
from data import Result
import data as dat
import preprocessing

folder_path = "/Users/sintes/Desktop/Martyna/PhD/chaines/2020-12-08_13h00m36s"

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
        im8 = (im16 * 0.99 * 2 ** 8 / max_int).astype("uint8")
    else:
        im8 = im16
    cv2.imwrite(new_name, im8)

def main(folder_path: str) -> None:
    print(folder_path)
    tmp = os.path.join(folder_path, "tmp")
    try:
        os.mkdir(tmp)
    except FileExistsError:
        pass

    image_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".tif")]
    image_list.sort()
    max_int = max_intensity_video(image_list)
    for i, im in enumerate(image_list):
        convert_16to8bits(im, i, max_int)

    image_list = [os.path.join(tmp, file) for file in os.listdir(tmp) if file.endswith(".png")]

    tracked_path = os.path.join(folder_path, "Figure","Tracked")
    shutil.rmtree(tracked_path, ignore_errors=True)
    os.makedirs(tracked_path)
    

    # Load configuration
    config = dat.Configuration()
    params = config.read_toml("/Users/sintes/Documents/Python/Chains/Chains/cfg.toml")
    # Data saver
    saver = Result(folder_path)

    # Set up detector

    visualisation_processed = True
    if visualisation_processed :
        processed_path = os.path.join(folder_path, "Figure", "Processed")
        shutil.rmtree(processed_path, ignore_errors=True)
        os.makedirs(processed_path)
        detector = ChainDetector(params, processed_path, visualisation=True)
    else:
        detector = ChainDetector(params, "", visualisation=False)
    background = preprocessing.get_background(image_list)
    cv2.imwrite(os.path.join(folder_path, "Figure/background.png"), background)
    detector.set_background(background)


    # Set up tracker
    tracker = Tracker(params, tracked_path) 
    tracker.set_params(params)
    tracker.set_detector(detector)

    camera = cv2.VideoCapture(
        "{}/Image%07d.png".format(tmp))
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

if __name__=="__main__":
    # parent_folder = "/Volumes/Chains/Chains"
    # folder_list: List[Tuple[str]] = [(os.path.join(parent_folder, f),) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))][0: 3]

    # pool = mp.Pool(mp.cpu_count() - 1)
    # pool.starmap_async(main, folder_list).get()
    # pool.close()
    main("/Users/sintes/Desktop/NASGuillaume/Chains/2023-10-06_13h10m14s")