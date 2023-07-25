#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import csv
import numpy as np

def calculate_centroids(binary_image):
    """
    Calculate the centroids of the objects in a binary image.

    Parameters:
        binary_image (np.ndarray): Input binarized image (0 for black, 1 for white).

    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates for each centroid.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the centroids of the contours
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    return centroids

def track_bacterial_chains(elongated_frames_folder):
    """
    Track the change of position of bacterial chains in each image of the elongated_frames folder.

    Parameters:
        elongated_frames_folder (str): Path to the folder containing elongated image frames.
    """
    # Create a CSV file to save the positions
    output_csv_file = os.path.join(elongated_frames_folder, 'bacterial_chain_positions.csv')
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'Chain', 'X Position', 'Y Position'])

        # Iterate over each image in the folder
        for filename in os.listdir(elongated_frames_folder):
            if filename.endswith('.png'):
                # Load the elongated image
                elongated_image_path = os.path.join(elongated_frames_folder, filename)
                elongated_image = cv2.imread(elongated_image_path, cv2.IMREAD_GRAYSCALE)

                # Binarize the elongated image
                _, binarized_image = cv2.threshold(elongated_image, 1, 255, cv2.THRESH_BINARY)

                # Calculate centroids of the bacterial chains
                centroids = calculate_centroids(binarized_image)

                # Save the positions in the CSV file
                for chain_id, (x, y) in enumerate(centroids, start=1):
                    csv_writer.writerow([filename, chain_id, x, y])

                print(f"Positions for {filename} saved.")

if __name__ == "__main__":
    elongated_frames_folder = 'F:\ACTIVE_NEW\elongated_frames'
    track_bacterial_chains(elongated_frames_folder)


# In[ ]:




