"""Preprocessing tools for the image analysis."""

import os
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.filters.thresholding import threshold_otsu
from typing import List

def estimate_background(image_sequence):
    """
    Estimate the background image using the minima method.

    Parameters:
        image_sequence (List[np.ndarray]): A list of grayscale image frames.

    Returns:
        np.ndarray: The estimated background image.
    """
    bg = np.copy(image_sequence[0])
    num_frames = len(image_sequence)
    rows, cols = bg.shape

    for t in range(1, num_frames):
        current_frame = image_sequence[t]
        for x in range(rows):
            for y in range(cols):
                v = current_frame[x, y]
                if v < bg[x, y]:
                    bg[x, y] = v

    return bg

def perform_background_removal(frames_folder, output_folder):
    """
    Perform background removal on a folder containing image frames.

    Parameters:
        frames_folder (str): Path to the folder containing the image frames.
        output_folder (str): Path to the desired output folder for background-removed images.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the frames and compute the background estimate
    image_list = [f for f in os.listdir(frames_folder) if f.endswith(".png") or f.endswith(".jpg")]
    num_frames = len(image_list)

    # Load the first frame to initialize the background
    background = cv2.imread(os.path.join(frames_folder, image_list[0]), cv2.IMREAD_GRAYSCALE)

    # Compute the background estimate
    for i in range(1, num_frames):
        filename = image_list[i]
        frame_path = os.path.join(frames_folder, filename)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        # Compute the background estimate if it's the first frame
        if i == 1:
            background = frame
        else:
            background = np.minimum(background, frame)

    # Save the background image
    background_path = os.path.join(output_folder, "background.png")
    cv2.imwrite(background_path, background)
    print(f"Background image saved as {background_path}")

    # Iterate over the frames again to perform background removal
    for i in range(num_frames):
        filename = image_list[i]
        frame_path = os.path.join(frames_folder, filename)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        # Compute the background-removed frame
        bg_removed_frame = frame - background

        # Save the background-removed image in the output folder
        output_filename = f"{filename.split('.')[0]}_background_removed.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, bg_removed_frame)
        print(f"Background-removed image saved as {output_path}")

def binarize(im: np.ndarray) -> np.ndarray:
    """
    Binarize an image using Otsu's method.

    Parameters:
        im (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Binarized image (0 for black, 1 for white).
    """
    im_gray = rgb2gray(im)
    threshold = threshold_otsu(im_gray)
    bin_im = (im_gray > threshold) * 1
    return bin_im

def perform_binarization_on_folder(input_folder, output_folder):
    """
    Binarize all images in a folder using Otsu's method.

    Parameters:
        input_folder (str): Path to the folder containing input grayscale images.
        output_folder (str): Path to the desired output folder for binarized images.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the input image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Binarize the image using Otsu's method
            binarized_image = binarize(image)

            # Save the binarized image in the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, binarized_image * 255)  # Scale values to 0-255 for saving as an 8-bit image
            print(f"Binarized image saved as {output_path}")

def elongate_objects(binarized_image, kernel_size=3):
    """
    Elongate the objects in a binarized image using dilation.

    Parameters:
        binarized_image (np.ndarray): Input binarized image (0 for black, 1 for white).
        kernel_size (int): Size of the square kernel for dilation. Default is 3.

    Returns:
        np.ndarray: The elongated image.
    """
    # Ensure that kernel_size is an integer
    kernel_size = int(kernel_size)

    # Create a square kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Dilate the binarized image
    iterations = 4  # Increase the number of iterations as desired
    dilated_image = cv2.dilate(binarized_image, kernel, iterations=iterations)

    return dilated_image


def calculate_actual_length(elongated_length, elongation_at_ends):
    """
    Calculate the actual chain length from the elongated chain length and elongation at the ends.

    Parameters:
        elongated_length (int): The total elongated chain length (number of white pixels).
        elongation_at_ends (int): The number of elongation pixels at the ends due to dilation.

    Returns:
        int: The actual chain length.
    """
    actual_length = elongated_length - 2 * elongation_at_ends
    return actual_length


def determine_actual_chain_lengths(input_image_path: str) -> List[int]:
    """
    Determine the actual chain lengths for all bacterial chains in the elongated image.

    Parameters:
        input_image_path (str): Path to the elongated image.

    Returns:
        List[int]: List of actual chain lengths for each bacterial chain in the image.
    """
    # Load the elongated image
    elongated_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Find contours in the elongated image
    contours, _ = cv2.findContours(elongated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the actual chain length for each contour
    actual_chain_lengths = []
    kernel_size = 3  # Adjust the value of kernel_size used in the dilation
    elongation_pixels_at_ends = 2 * (kernel_size - 1)

    for contour in contours:
        # Calculate elongated chain length for each contour (number of white pixels)
        elongated_chain_length_pixels = cv2.contourArea(contour)

        # Calculate actual chain length for the contour
        actual_chain_length = calculate_actual_length(elongated_chain_length_pixels, elongation_pixels_at_ends)
        actual_chain_lengths.append(actual_chain_length)

    return actual_chain_lengths


def calculate_original_chain_lengths(elongated_image_path: str) -> list:
    """
    Calculate the original chain lengths for all bacterial chains in the elongated image.

    Parameters:
        elongated_image_path (str): The file path to the elongated image.

    Returns:
        list: A list of original chain lengths (in micrometers) for each bacterial chain in the image.
    """
    # Load the elongated image
    elongated_image = cv2.imread(elongated_image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the elongated image
    _, binarized_image = cv2.threshold(elongated_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the elongated image
    contours, _ = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate original chain lengths for all bacterial chains in the elongated image
    original_chain_lengths = []
    for contour in contours:
        chain_length_pixels = cv2.arcLength(contour, True)
        chain_length_micrometers = chain_length_pixels * 0.2  # Assuming 1 pixel corresponds to 0.2 micrometers
        original_chain_lengths.append(chain_length_micrometers)

    return original_chain_lengths


def perform_elongation_on_folder(binarized_folder: str, elongated_frames_folder: str):
    """
    Perform elongation of objects in the binarized images within the specified folder.

    The function elongates the objects in each binarized image using dilation and saves the elongated
    images in the specified output folder.

    Parameters:
        binarized_folder (str): The path to the folder containing binarized images (PNG format).
        elongated_frames_folder (str): The path to the output folder for saving the elongated images.

    Returns:
        None
    """
    # Create the output folder if it doesn't exist
    os.makedirs(elongated_frames_folder, exist_ok=True)

    # Iterate over each image in the binarized folder
    for filename in os.listdir(binarized_folder):
        if filename.endswith('.png'):
            # Load the binarized image
            binarized_image_path = os.path.join(binarized_folder, filename)
            binarized_image = cv2.imread(binarized_image_path, cv2.IMREAD_GRAYSCALE)

            # Elongate the objects in the binarized image
            elongated_image = elongate_objects(binarized_image)

            # Save the elongated image in the output folder
            elongated_image_path = os.path.join(elongated_frames_folder, filename)
            cv2.imwrite(elongated_image_path, elongated_image)
            print(f"Elongated image saved as {elongated_image_path}")
