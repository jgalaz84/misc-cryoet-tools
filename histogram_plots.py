#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya, 03/2023; last update 07/2024

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
import os

def read_data(file_path):
    """ Read data from a text file. Assumes there's one column of numeric data. """
    return pd.read_csv(file_path, header=None).iloc[:, 0]

def plot_histogram(data, label, ax):
    """ Plot histogram and the mean and median lines. """
    n, bins, patches = ax.hist(data, bins=30, alpha=0.7, label=f'{label} - Hist')
    mean = data.mean()
    median = data.median()
    ax.axvline(mean, color=patches[0].get_facecolor(), linestyle='-', linewidth=2, label=f'{label} - Mean')
    ax.axvline(median, color=patches[0].get_facecolor(), linestyle='--', linewidth=2, label=f'{label} - Median')

def create_output_dir(base_dir="plots"):
    """ Create an output directory for saving figures. Increment the suffix number if dir already exists. """
    i = 0
    while True:
        new_dir = f"{base_dir}_{str(i).zfill(2)}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        i += 1

def main(args):
    file_paths = args.input.split(',')
    labels = args.labels.split(',')
    
    # Prepare the output directory
    output_dir = create_output_dir()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    datasets = []

    # Read and plot data for each file
    for file_path, label in zip(file_paths, labels):
        data = read_data(file_path)
        plot_histogram(data, label, ax)
        datasets.append(data)
    
    # Perform statistical significance tests
    combinations = [(0, 1), (1, 2), (0, 2)] if len(datasets) > 2 else [(0, 1)]
    for (i, j) in combinations:
        stat, p_value = ttest_ind(datasets[i], datasets[j], equal_var=False)
        print(f'Statistical test between {labels[i]} and {labels[j]}: p-value = {p_value:.4f}')
    
    # Formatting the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title('Histograms with Mean and Median')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    # Save the plot to the output directory
    plot_path = os.path.join(output_dir, 'histograms.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved to {plot_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot histograms from CSV files.')
    parser.add_argument('--input', type=str, help='Comma-separated list of file paths.')
    parser.add_argument('--labels', type=str, help='Comma-separated list of labels for the histograms.')
    args = parser.parse_args()
    main(args)
