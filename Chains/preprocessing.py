"""Preprocessing tools for the image analysis."""

from typing import List, Tuple

import cv2
import numpy as np
from skimage.filters.thresholding import threshold_otsu
from skimage import morphology


def get_background(image_sequence: List[str]) -> np.ndarray:
    """
    Estimate the background image using the minima method.

    Parameters:
        image_sequence (List[str]): A list of grayscale image frames.

    Returns:
        np.ndarray: The estimated background image.
    """
    background = cv2.imread(image_sequence[0], cv2.IMREAD_UNCHANGED)

    for im_name in image_sequence:
        current_frame = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
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
    bin_im = (im > threshold)
    return bin_im

def remove_small_objects(bin_image: np.ndarray) -> np.ndarray:
    """Remove the small objects in a binerized image."""
    return morphology.remove_small_objects(bin_image, min_size=50, connectivity=2)

def elongate_objects(binarized_image: np.ndarray, nb_iter: int = 2, kernel_size: int = 5) -> np.ndarray:
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
    dilated_image = binarized_image.copy()
    for _ in range(nb_iter):
        dilated_image = morphology.binary_dilation(dilated_image, kernel)
    return dilated_image

def contour_on_the_side(contour: List[List[List[int]]], im_shape: Tuple[int, int]) -> bool:
    """Detect if a contour is touching the side of the image."""
    for i in contour:
        point = i[0]
        y = point[0]
        x = point[1]
        
        if x == 0 or x == im_shape[0] -1:
            return True
        if y == 0 or y == im_shape[1] -1:
            return True
    return False


# def get_elongation(kernel_size: int = 3, iterations: int = 4) -> float:
#     """
#     Calculate the actual chain length from the elongated chain length and elongation at the ends.
#     """
#     return 2 * iterations * (kernel_size - 1) 