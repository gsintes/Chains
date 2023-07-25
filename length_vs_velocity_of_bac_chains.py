#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pandas as pd
import matplotlib.pyplot as plt

def calculate_chain_velocity(x1, y1, x2, y2, time_interval=0.1):
    # Calculate the velocity between two chain positions
    velocity = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / time_interval
    return velocity

def main():
    # Input directory
    input_directory = 'F:\ACTIVE_NEW\CSV'  # Replace with the path to your output folder containing the CSV files

    # Read chain lengths and chain positions from the CSV files using pandas
    chain_lengths_csv = os.path.join(input_directory, 'bacterial_chain_lengths.csv')
    chain_lengths_df = pd.read_csv(chain_lengths_csv)

    chain_positions_csv = os.path.join(input_directory, 'bacterial_chain_positions.csv')
    chain_positions_df = pd.read_csv(chain_positions_csv)

    # Merge chain lengths and chain positions dataframes
    merged_df = pd.merge(chain_lengths_df, chain_positions_df, on=['Image', 'Chain Number'])

    # Prepare data for the scatter plot
    chain_lengths_list = []
    chain_velocities_list = []

    for _, chain_group in merged_df.groupby('Chain Number'):
        # Sort the chain group by frame number
        sorted_chain_group = chain_group.sort_values('Image')

        # Calculate velocities for each pair of consecutive frames
        x_values = sorted_chain_group['X Position (micrometers)'].values
        y_values = sorted_chain_group['Y Position (micrometers)'].values
        velocities = [calculate_chain_velocity(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(x_values[:-1], y_values[:-1], x_values[1:], y_values[1:])]

        # Calculate the mean velocity for the chain
        mean_velocity = sum(velocities) / len(velocities) if len(velocities) > 0 else 0.0

        # Append the chain length and mean velocity to the lists
        chain_lengths_list.append(sorted_chain_group['Chain Length (micrometers)'].iloc[0])
        chain_velocities_list.append(mean_velocity)

    # Plot the scatter plot
    plt.scatter(chain_lengths_list, chain_velocities_list)
    plt.xlabel('Bacterial Chain Length (micrometers)')
    plt.ylabel('Mean Chain Velocity (micrometers per time interval)')
    plt.title('Bacterial Chain Length vs. Mean Chain Velocity')
    plt.show()

if __name__ == "__main__":
    main()


# In[ ]:




