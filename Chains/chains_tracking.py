
import os
from typing import List, Tuple
import shutil
import multiprocessing as mp
import json
from datetime import datetime

import cv2
import numpy as np

from chains_detector import ChainDetector
from tracker import Tracker
from data import Result
import data as dat
import preprocessing

def main(folder_path: str) -> str:
    exp_name = folder_path.split("/")[-1]

    image_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".tif")]
    image_list.sort()
    param_file = os.path.join(folder_path, "params.json")
    try:
        f = open(param_file)
        parsed_params = json.load(f)
        max_int = parsed_params["maxint"]
    except FileNotFoundError:
        max_int = int(preprocessing.max_intensity_video(image_list))
        with open(param_file, 'w') as f:
            f.write(json.dumps({"maxint": max_int}))
    
    
    tracked_path = os.path.join(folder_path, "Figure","Tracked")
    try:
        os.mkdir(tracked_path)
    except FileExistsError:
        pass
    

    # Load configuration
    config = dat.Configuration()
    params = config.read_toml(os.path.join(os.getcwd(),"cfg.toml"))
    # Data saver
    saver = Result(folder_path)

    # Set up detector

    visualisation_processed = False
    if visualisation_processed :
        processed_path = os.path.join(folder_path, "Figure", "Processed")
        shutil.rmtree(processed_path, ignore_errors=True)
        os.makedirs(processed_path)
        detector = ChainDetector(params, processed_path, visualisation=True)
    else:
        detector = ChainDetector(params, "", visualisation=False)

    bg_path = os.path.join(folder_path, "Figure/background.png")
    if os.path.isfile(bg_path):
        background = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    else:
        background = preprocessing.get_background(image_list, max_int)
        cv2.imwrite(bg_path, background)
    detector.set_background(background)


    # Set up tracker
    tracker = Tracker(params, tracked_path) 
    tracker.set_params(params)
    tracker.set_detector(detector)

    im_data = tracker.initialize(preprocessing.convert_16to8bits(image_list[0], max_int))
    saver.add_data(im_data)

    for im in image_list[1:]:
        frame = preprocessing.convert_16to8bits(im, max_int)
        im_data = tracker.process(frame)
        saver.add_data(im_data)
    return f"{exp_name} done at {datetime.now()}\n"

if __name__=="__main__":
    parent_folder = "/run/user/1000/gvfs/afp-volume:host=Suspension_Lab.local,volume=Guillaume/Chains"
    log_file = os.path.join(parent_folder, "log.txt")

    # parent_folder = "/Users/sintes/Desktop/NASGuillaume/Chains/"
    folder_list: List[Tuple[str]] = [(os.path.join(parent_folder, f),) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]

    # pool = mp.Pool(mp.cpu_count() - 1)
    # pool.starmap_async(main, folder_list).get()
    # pool.close()
    for f in folder_list:
        log = main(f[0])
        with open(log_file, 'a') as file:
            file.write(log)
    # main("/Users/sintes/Desktop/NASGuillaume/Chains/2023-10-06_13h10m14s")
