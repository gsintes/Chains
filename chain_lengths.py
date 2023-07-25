"""Detect the chains and there length on an image."""

import os
import preprocessing
import csv

def main():
    # Input and output directories
    input_directory = 'C:/AS'  # Replace with the path to your input folder containing the binarized images
    output_directory = 'F:\G'  # Replace with the path to your desired output folder for CSV files

    # Create subfolders for background removal, binarization, elongation, and CSV output
    bg_removal_folder = os.path.join(output_directory, 'background_removal')
    binarization_folder = os.path.join(output_directory, 'binarization')
    elongated_frames_folder = os.path.join(output_directory, 'elongated_frames')
    os.makedirs(bg_removal_folder, exist_ok=True)
    os.makedirs(binarization_folder, exist_ok=True)
    os.makedirs(elongated_frames_folder, exist_ok=True)

    # Perform background removal on the input folder and save the results in the output folder
    preprocessing.perform_background_removal(input_directory, bg_removal_folder)

    # Perform binarization on the background removal folder and save the binarized images in the binarization folder
    preprocessing.perform_binarization_on_folder(bg_removal_folder, binarization_folder)

    # Perform elongation of chains on the binarization folder and save the elongated images in the elongated frames folder
    preprocessing.perform_elongation_on_folder(binarization_folder, elongated_frames_folder)

    # Process each elongated image in the elongated frames folder
    results = []
    for filename in os.listdir(elongated_frames_folder):
        if filename.endswith('.png'):
            # Get the path to the elongated image
            elongated_image_path = os.path.join(elongated_frames_folder, filename)

            # Determine original chain lengths for all bacterial chains in the elongated image
            original_chain_lengths = preprocessing.calculate_original_chain_lengths(elongated_image_path)

            # Append the results to the list
            for chain_number, chain_length in enumerate(original_chain_lengths, start=1):
                results.append([filename, chain_number, chain_length])

    # Write the results to a CSV file
    output_csv_file = os.path.join(output_directory, 'bacterial_chain_lengths.csv')
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'Chain Number', 'Chain Length (micrometers)'])
        csv_writer.writerows(results)

if __name__ == "__main__":
    main()