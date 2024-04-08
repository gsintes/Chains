"""Preprocessing tools for the image analysis."""

from typing import List, Tuple
import time

import cv2
import numpy as np
from skimage.filters.thresholding import threshold_otsu
from skimage import morphology
from skimage.filters import gaussian
from skimage.measure import label, regionprops, regionprops_table

def timeit(func):
    # @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} s')
        return result
    return timeit_wrapper

def max_intensity_video(image_list: List[str]) -> int:
    """Detect the maximum intensity in a video."""
    max_int = 0
    for im_name in image_list:
        im = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
        max_int = max(max_int, np.amax(im))
    return max_int


def convert_16to8bits(image: str, max_int: int) -> np.ndarray:
    """Convert 16bit image to 8bit and store it in a temp folder."""
    im16 = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if im16.dtype == "uint16":
        im8 = (im16 * 0.99 * 2 ** 8 / max_int).astype("uint8")
    else:
        im8 = im16
    return im8


def get_background(image_sequence: List[str], max_int: int) -> np.ndarray:
    """
    Estimate the background image using the minima method.

    Parameters:
        image_sequence (List[str]): A list of grayscale image frames.

    Returns:
        np.ndarray: The estimated background image.
    """
    background = convert_16to8bits(image_sequence[0], max_int)

    for im_name in image_sequence:
        current_frame = convert_16to8bits(im_name, max_int)
        background = np.minimum(background, current_frame)
    return background

def remove_slowy_varying(image: np.ndarray) -> np.ndarray:
    """Remove slowy image."""
    image = image.astype("float64")
    blur = gaussian(image, sigma=10)
    return image - blur

def binarize(im: np.ndarray) -> np.ndarray:
    """
    Binarize an image using Otsu's method.

    Parameters:
        im (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Binarized image (0 for black, 1 for white).
    """
    threshold = threshold_otsu(im)
    threshold = max(15, threshold)
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
    label_img = label(binarized_image)
    regions = regionprops(label_img)
    binarized_image = binarized_image.astype("int8") * 255
    for props in regions:
        centroid_y = props.centroid[0]
        centroid_x = props.centroid[1]
        orientation = props.orientation
        long_axis = props.axis_major_length
        new_long = 4 + 0.5 * long_axis
        minor_axis = max(1, int(0.5 * props.axis_minor_length))
        c, s = np.cos(-orientation), np.sin(-orientation)
        R = np.array(((c, -s), (s, c)))      
        c1 = np.floor(np.array([centroid_x, centroid_y]) + R @ (0, - new_long)).astype("int64")
        c4 = np.floor(np.array([centroid_x, centroid_y]) + R @ (0,new_long)).astype("int64")
        binarized_image = cv2.line(binarized_image, c1, c4, 255, minor_axis)
    return binarized_image 

def contour_on_the_side(contour: List[List[List[int]]], im_shape: Tuple[int, ...]) -> bool:
    """Detect if a contour is touching the side of the image."""
    for i in contour:
        point = i[0]
        y = point[0]
        x = point[1]
        
        if x == 0 or x == im_shape[0] - 1:
            return True
        if y == 0 or y == im_shape[1] - 1:
            return True
    return False
