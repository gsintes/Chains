"""Program to run tracking in chain of bacteria."""

import os
import shutil
import json
from datetime import datetime

import cv2

from chains_detector import ChainDetector
from kalman_tracker import ObjectTracker
from data import Result
import data as dat
import preprocessing

def main(folder_path: str) -> str:
    exp_name = folder_path.split("/")[-1]
    print(exp_name)

    image_list = [os.path.join(folder_path, file) for file
                in os.listdir(folder_path) if file.endswith(".tif")]
    image_list.sort()
    param_file = os.path.join(folder_path, "params.json")
    try:
        with open(param_file, "r") as f:
            parsed_params = json.load(f)
            max_int = parsed_params["maxint"]
    except FileNotFoundError:
        max_int = int(preprocessing.max_intensity_video(image_list))
        with open(param_file, 'w') as f:
            f.write(json.dumps({"maxint": max_int}))

    tracked_path = os.path.join(folder_path, "Figure","Tracked")
    try:
        os.makedirs(tracked_path)
    except FileExistsError:
        pass

    try:
        os.makedirs(os.path.join(folder_path,"Tracking_Result"))
    except FileExistsError:
        shutil.rmtree(os.path.join(folder_path,"Tracking_Result"))
        os.makedirs(os.path.join(folder_path,"Tracking_Result"))

    # Load configuration
    config = dat.Configuration()
    params = config.read_toml(os.path.join(os.getcwd(),"cfg.toml"))
    # Data saver
    saver = Result(os.getcwd())

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
        background = preprocessing.get_background(image_list[0:100], max_int)
        cv2.imwrite(bg_path, background)
    detector.set_background(background)


    # Set up tracker
    tracker = ObjectTracker(params, detector, tracked_path)

    im_data = tracker.initialize(preprocessing.convert_16to8bits(image_list[0], max_int))
    saver.add_data(im_data)

    count = 0
    for i, im in enumerate(image_list[1:]):
        per = 100 * i / len(image_list)
        if per > count + 5:
            print(f"{per:.2f}2%")
            count += 5
        frame = preprocessing.convert_16to8bits(im, max_int)
        im_data = tracker.process(frame)
        saver.add_data(im_data)
    shutil.move(os.path.join(os.getcwd(), "tracking.db"), os.path.join(folder_path,"Tracking_Result"))
    return f"{exp_name} done at {datetime.now()}\n"

if __name__=="__main__":
    # parent_folder = "/Users/sintes/Desktop/NASGuillaume/5min/2024-03-26_14h15m09s"
    # log_file = os.path.join(parent_folder, "log.txt")

    # with open(log_file, 'w') as file:
    #     file.write("Tracking code \n")

    # folder_list: List[Tuple[str]] = [(os.path.join(parent_folder, f),) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]
    # folder_list.sort()

    # for f in folder_list:
    #     try:
    #         log = main(f[0])
    #         with open(log_file, 'a') as file:
    #             file.write(log)
    #     except Exception as e:
    #         with open(log_file, 'a') as file:
    #             exp_name = f.split("/")[-1]
    #             file.write(f"{exp_name} error at {datetime.now()}: {e.__repr__}\n")
    folder = "/Volumes/Chains/ChainFormation/2024-05-09_11h28m54s"
    main(folder)
