"""Preprocessing tools for the image analysis."""

from typing import List

import cv2
import numpy as np
import matplotlib.image as mpim
from skimage.filters.thresholding import threshold_otsu


def get_background(image_sequence: List[str]) -> np.ndarray:
    """
    Estimate the background image using the minima method.

    Parameters:
        image_sequence (List[str]): A list of grayscale image frames.

    Returns:
        np.ndarray: The estimated background image.
    """
    background = mpim.imread(image_sequence[0])

    for im_name in image_sequence:
        current_frame = mpim.imread(im_name)
        background = np.minimum(background, current_frame)
    return background

def binarize(im: np.ndarray) -> np.ndarray:
    """
    Binarize an image using Otsu's method.

    Parameters:
        im (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Binarized image (0 for black, 1 for white).
    """
    threshold = threshold_otsu(im)
    bin_im = (im > threshold) * 1
    return bin_im

def elongate_objects(binarized_image: np.ndarray, kernel_size: int = 3, iterations: int = 4) -> np.ndarray:
    """
    Elongate the objects in a binarized image using dilation to fill the gaps in the chains.

    Parameters:
        binarized_image (np.ndarray): Input binarized image.
        kernel_size (int): Size of the square kernel for dilation. Default is 3.
        iterations (int): Number of iteration of the dilation. Default is 4.

    Returns:
        np.ndarray: The elongated image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_image = cv2.dilate(binarized_image, kernel, iterations=iterations)

    return dilated_image


# def get_elegonation(kernel_size: int = 3, iterations: int = 4) -> float:
#     """
#     Calculate the actual chain length from the elongated chain length and elongation at the ends.
#     """
#     return 2 * iterations * (kernel_size - 1) 