"""Track the chains."""

from typing import List
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import preprocessing

def pretreatment(image: np.ndarray, background: np.ndarray, i: int, fig_folder:str, visualization: bool) -> np.ndarray:
    """Performs the pretreatment of the image before tracking analysis."""
    bg_removed = image - background
    bin_image = preprocessing.binarize(bg_removed)
    cleaned = preprocessing.remove_small_objects(bin_image)
    elongated = preprocessing.elongate_objects(cleaned)

    if visualization:
        mpim.imsave(os.path.join(fig_folder, f"Processed/processed_image_{i}.png"), elongated, cmap="gray")
    return elongated

def main(folder: str, visualization: bool = True) -> None:
    """Perform chain tracking."""
    fig_folder = os.path.join(folder, "Figures/")
    if os.path.isdir(fig_folder):
        shutil.rmtree(fig_folder)
    os.makedirs(fig_folder)
    os.makedirs(os.path.join(fig_folder, "Processed"))
    os.makedirs(os.path.join(fig_folder, "Track_verif"))
    list_image = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".tif") and not(file.startswith("."))]
    list_image.sort()
    background = preprocessing.get_background(list_image)
    
    for i, im_name in enumerate(list_image[0:10]):
        im = mpim.imread(im_name)
        processed = pretreatment(im, background, i, fig_folder, visualization=True)

    if visualization:
        mpim.imsave(os.path.join(fig_folder, "background.png"), background, cmap="gray")

if __name__ == "__main__":
    folder = "/Users/sintes/Desktop/ImageSeq"
    main(folder, visualization=True)