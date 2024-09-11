"""Program to run tracking in chain of bacteria."""

import os
from typing import List
import shutil
from datetime import datetime

import cv2

from chains_detector import ChainDetector
from kalman_tracker import ObjectTracker
from data import Result
import data as dat
import preprocessing

import time
def main(folder_path: str) -> str:
    """Run tracking on a movie."""
    exp_name = folder_path.split("/")[-1]
    print(exp_name)
    window_size = 100
    block_tracker = BlockTracker(folder_path, window_size)

    return f"{exp_name} done at {datetime.now()}\n"

class BlockTracker:
    """Run tracking on a movie by making block of images."""
    def __init__(self, folder_path: str, block_size: int) -> None:
        self.folder_path = folder_path
        self.block_size = block_size

        self.image_list = [os.path.join(folder_path, file) for file
                    in os.listdir(folder_path) if file.endswith(".tif")]
        self.image_list.sort()

        self.tracked_path = os.path.join(self.folder_path, "Figure","Tracked")
        try:
            os.makedirs(self.tracked_path)
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join(self.folder_path, "Figure","Background"))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join(self.folder_path, "Tracking_Result"))
        except FileExistsError:
            shutil.rmtree(os.path.join(self.folder_path, "Tracking_Result"))
            os.makedirs(os.path.join(self.folder_path, "Tracking_Result"))
        self.block_tracking(0)
        self.block_tracking(1)

    def block_tracking(self, block_number: int) -> str:
        """Run tracking on a block of images."""
        im_list = self.image_list[block_number * self.block_size :(block_number + 1) * self.block_size]
        max_int = int(preprocessing.max_intensity_video(im_list))

        # Load configuration
        config = dat.Configuration()
        params = config.read_toml(os.path.join(os.getcwd(),"cfg.toml"))
        # Data saver
        saver = Result(os.getcwd(), block_number)

        # Set up detector
        detector = ChainDetector(params, "", visualisation=False)

        bg_path = os.path.join(self.folder_path, f"Figure/Background/background_{block_number}.png")
        if os.path.isfile(bg_path):
            background = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        else:
            background = preprocessing.get_background(im_list, max_int)
            cv2.imwrite(bg_path, background)
        detector.set_background(background)


        # # Set up tracker
        tracker = ObjectTracker(params, detector, self.block_size * block_number, self.tracked_path)

        im_data = tracker.initialize(preprocessing.convert_16to8bits(im_list[0], max_int))
        saver.add_data(im_data)

        for i, im in enumerate(im_list[1:]):
            frame = preprocessing.convert_16to8bits(im, max_int)
            im_data = tracker.process(frame)
            saver.add_data(im_data)
        shutil.move(os.path.join(os.getcwd(), f"tracking_{block_number}.db"), os.path.join(self.folder_path,"Tracking_Result"))


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
    folder = "/Users/sintes/Desktop/2023-10-31_11h15m10s"
    main(folder)
