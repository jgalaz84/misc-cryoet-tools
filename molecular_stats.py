#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2024; last modification: Sep/03/2024

import h5py
import argparse
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import kurtosis, ttest_ind, mannwhitneyu, shapiro
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import sys
import os
import logging
import traceback
import time

def list_hdf5_paths(hdf5_file):
    """ Lists all paths in an HDF5 file """
    paths = []
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            paths.append(name)
    with h5py.File(hdf5_file, 'r') as file:
        file.visititems(visitor_func)
    return paths

def load_and_process_data(file_path):
    """Helper function to load 3D image data (or a stack of 3D volumes) and calculate metrics."""
    
    print(f"Attempting to load data from {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            paths = list_hdf5_paths(file_path)
            print(f"Available dataset paths in {file_path}: {paths}")
            image_dataset_paths = [path for path in paths if path.endswith('/image')]
            if not image_dataset_paths:
                raise KeyError("No datasets with '/image' found in the file.")

            image_stack = np.array([file[path][:] for path in image_dataset_paths])
            print(f"(load_and_process_data) Loaded data shape: {image_stack.shape}")
    
    elif file_extension == '.mrc':
        with mrcfile.open(file_path, permissive=True) as mrc:
            image_stack = np.expand_dims(mrc.data, axis=0)
            print(f"(load_and_process_data) Loaded MRC data shape: {image_stack.shape}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    num_volumes = image_stack.shape[0]
    print(f"A total of N={num_volumes} 3D volumes found in stack={file_path}")
    
    global_metrics = {
        'kurtosis': [], 'entropy': [], 'contrast': [], 'homogeneity': [],
        'energy': [], 'correlation': [], 'mean': [], 'std_dev': []
    }
    global_extreme_slices = {key: {'min': (None, float('inf'), None), 'max': (None, float('-inf'), None)} for key in global_metrics}

    for vol_idx in range(num_volumes):
        image_3d = image_stack[vol_idx]
        print(f"Calculating stats for image n={vol_idx+1}/{num_volumes} in stack={file_path}")

        metrics, extreme_slices = calculate_metrics_for_images(image_3d)

        for key in global_metrics:
            global_metrics[key].extend(metrics[key])

        for key in global_metrics:
            if extreme_slices[key]['min'][1] < global_extreme_slices[key]['min'][1]:
                global_extreme_slices[key]['min'] = (vol_idx, extreme_slices[key]['min'][1], extreme_slices[key]['min'][0])
            if extreme_slices[key]['max'][1] > global_extreme_slices[key]['max'][1]:
                global_extreme_slices[key]['max'] = (vol_idx, extreme_slices[key]['max'][1], extreme_slices[key]['max'][0])

    final_extreme_slices = {key: {'min_slice': image_stack[global_extreme_slices[key]['min'][0]][global_extreme_slices[key]['min'][2]],
                                  'max_slice': image_stack[global_extreme_slices[key]['max'][0]][global_extreme_slices[key]['max'][2]]}
                            for key in global_metrics}

    # Return image_stack in addition to the metrics and extreme slices
    return image_stack, global_metrics, final_extreme_slices


def setup_logger(output_dir):
    """Setup a logger with a file handler to log to a specified directory."""
    logger = logging.getLogger('molstats_logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, 'molstats.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()  # Console handler to output to the terminal as well.
    ch.setLevel(logging.ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def create_output_directory(base_dir):
    """Create and return a numbered output directory."""
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    return output_dir

def calculate_metrics_for_images(image_3d):
    print("\nCalculating metrics for loaded image data...")  # Print statement for clarity
    metrics = {
        'kurtosis': [], 'entropy': [], 'contrast': [], 'homogeneity': [],
        'energy': [], 'correlation': [], 'mean': [], 'std_dev': []
    }
    
    # Process the slices
    extreme_slices = process_slices(image_3d, metrics)
    return metrics, extreme_slices


def process_slices(image_3d, metrics):
    """Process all slices from a 3D volume and calculate statistical metrics."""
    print(f'\nAnalyzing slices...')
    
    extremes = {key: {'min': (None, float('inf')), 'max': (None, float('-inf'))} for key in metrics}
    
    num_slices = image_3d.shape[0]  # Assuming the first dimension corresponds to the slices
    
    for idx in range(num_slices):
        slice_2d = image_3d[idx, :, :]  # Extract the 2D slice from the 3D volume (z-axis)
        mean = np.mean(slice_2d)
        sigma = np.std(slice_2d)

        if sigma > 0:
            # Normalize the 2D slice (assuming float32/float64 data requires scaling)
            slice_normalized = (slice_2d * 255).astype(np.uint8) if slice_2d.dtype in [np.float32, np.float64] else slice_2d
            
            # Calculate GLCM for the 2D slice
            glcm = graycomatrix(slice_normalized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
            
            feature_values = {
                'kurtosis': kurtosis(slice_normalized.ravel(), fisher=True),
                'entropy': shannon_entropy(slice_normalized),
                'contrast': graycoprops(glcm, 'contrast')[0].mean(),
                'homogeneity': graycoprops(glcm, 'homogeneity')[0].mean(),
                'energy': graycoprops(glcm, 'energy')[0].mean(),
                'correlation': graycoprops(glcm, 'correlation')[0].mean(),
                'mean': mean,
                'std_dev': sigma
            }
            
            # Append metrics and check for extremes
            for key, value in feature_values.items():
                metrics[key].append(value)
                if value < extremes[key]['min'][1]:
                    extremes[key]['min'] = (idx, value)
                if value > extremes[key]['max'][1]:
                    extremes[key]['max'] = (idx, value)
        else:
            sys.stdout.write(f"Skipping slice {idx} due to zero standard deviation")
            sys.stdout.flush()

        # In-place printing with slice index starting from 1
        sys.stdout.write(f"\rProcessing slice {idx+1}/{num_slices}, slice shape: {slice_2d.shape}")
        sys.stdout.flush()
    
    print()  # Move to the next line after processing all slices
    return extremes


def is_normal(data, alpha=0.05):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    stat, p_value = shapiro(data)
    return p_value > alpha  # Returns True if data is normal (p-value > alpha)



def plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2, metric_keys, filename, data_labels=None):
    print("\nCreating violin plots for", filename)
    valid_metrics = []
    p_values = []
    significance_labels = []

    # Set default data labels if none provided
    if data_labels is None:
        data_labels = ['Dataset 1', 'Dataset 2']
    else:
        data_labels = data_labels.split(',')

    # Determine valid metrics and calculate stats (mean, median, stderr)
    stats_summary1 = {}
    stats_summary2 = {}
    
    for key in metric_keys:
        data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])

        # Skip if data contains NaN values
        if np.isnan(data1).all() or np.isnan(data2).all():
            print(f"Skipping {key} due to invalid data")
            continue

        # Check normality and select the appropriate test
        normal1 = is_normal(data1)
        normal2 = is_normal(data2)

        if normal1 and normal2:
            stat, p_value = ttest_ind(data1, data2)
        else:
            stat, p_value = mannwhitneyu(data1, data2)

        # Skip if p-value is NaN
        if np.isnan(p_value):
            print(f"Skipping {key} due to NaN p-value")
            continue

        valid_metrics.append(key)
        p_values.append(p_value)

        # Determine significance label
        if p_value < 0.001:
            significance_labels.append("***")
        elif p_value < 0.01:
            significance_labels.append("**")
        elif p_value < 0.05:
            significance_labels.append("*")
        else:
            significance_labels.append("ns")

        # Calculate mean, median, and standard error for dataset1 and dataset2
        mean1, median1, stderr1 = np.mean(data1), np.median(data1), np.std(data1) / np.sqrt(len(data1))
        mean2, median2, stderr2 = np.mean(data2), np.median(data2), np.std(data2) / np.sqrt(len(data2))
        
        stats_summary1[key] = (mean1, median1, stderr1)
        stats_summary2[key] = (mean2, median2, stderr2)

    num_metrics = len(valid_metrics)

    # If no valid metrics, print a warning and exit the function
    if num_metrics == 0:
        print("No valid metrics to plot.")
        return

    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 8))  # Increase height for better layout

    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Color palette for distinction

    # Plot valid metrics
    for i, key in enumerate(valid_metrics):
        ax = axes[i] if num_metrics > 1 else axes  # Handle case of single subplot
        data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])

        # Adjusting y-limits to start from 0 or the minimum value PLUS PADDING
        min_val1 = np.min(data1)
        max_val1 = np.max(data1)
        min_val2 = np.min(data2)
        max_val2 = np.max(data2)

        global_min = min(min_val1, min_val2)
        global_max = max(max_val1, max_val2)

        datarange = global_max - global_min
        global_min_padded = global_min - datarange * 0.1
        global_max_padded = global_max + datarange * 0.1
        ax.set_ylim(global_min_padded, global_max_padded)

        # Plot violin plot for Dataset 1
        parts1 = ax.violinplot(data1, positions=[1], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts1['bodies']:
            pc.set_facecolor(color_palette[i % len(color_palette)])
            pc.set_alpha(0.3)  # Set transparency for violin plot overlay
            pc.set_edgecolor('black')

        # Overlay scatter plot of data points for Dataset 1
        ax.scatter(np.full_like(data1, 1), data1, color=color_palette[i % len(color_palette)], alpha=0.7, edgecolor='k')

        # Custom additions for quartiles, medians, and extrema for Dataset 1
        quartile1, med1, quartile3 = np.percentile(data1, [25, 50, 75])
        mean1 = np.mean(data1)

        ax.scatter(1, med1, color='red', zorder=3)  # Median (Red)
        ax.scatter(1, mean1, color='white', zorder=3)  # Mean (White)
        ax.vlines(1, min_val1, max_val1, color='black', linestyle='-', lw=2)  # Min-Max range (Black)
        ax.vlines(1, quartile1, quartile3, color='black', linestyle='-', lw=5)  # Interquartile range (Black)

        # Label the distribution as Gaussian or Non-Gaussian for Dataset 1
        label1 = 'Gaussian' if normal1 else 'Non-Gaussian'
        ax.text(1, global_max_padded, label1, ha='center', va='bottom', fontsize=10, color='black')

        # Plot violin plot for Dataset 2
        parts2 = ax.violinplot(data2, positions=[2], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts2['bodies']:
            pc.set_facecolor(color_palette[i % len(color_palette)])
            pc.set_alpha(0.3)  # Set transparency for violin plot overlay
            pc.set_edgecolor('black')

        # Overlay scatter plot of data points for Dataset 2
        ax.scatter(np.full_like(data2, 2), data2, color=color_palette[i % len(color_palette)], alpha=0.7, edgecolor='k')

        # Custom additions for quartiles, medians, and extrema for Dataset 2
        quartile1, med2, quartile3 = np.percentile(data2, [25, 50, 75])
        mean2 = np.mean(data2)

        ax.scatter(2, med2, color='red', zorder=3)  # Median (Red)
        ax.scatter(2, mean2, color='white', zorder=3)  # Mean (White)
        ax.vlines(2, min_val2, max_val2, color='black', linestyle='-', lw=2)  # Min-Max range (Black)
        ax.vlines(2, quartile1, quartile3, color='black', linestyle='-', lw=5)  # Interquartile range (Black)

        # Label the distribution as Gaussian or Non-Gaussian for Dataset 2
        label2 = 'Gaussian' if normal2 else 'Non-Gaussian'
        ax.text(2, global_max_padded, label2, ha='center', va='bottom', fontsize=10, color='black')

        # Display numerical p-value, mean, and median
        N1, N2 = len(data1), len(data2)
        ax.set_title(f'{key} (p={p_values[i]:.6f}, {significance_labels[i]})\nN={N1}, N={N2}\nMean1={mean1:.6f}, Median1={med1:.6f}\nMean2={mean2:.6f}, Median2={med2:.6f}', fontsize=10, pad=30)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(data_labels)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Saved {filename} successfully.")

    # Save mean, median, and standard error to text files
    for label, stats_summary in zip(data_labels, [stats_summary1, stats_summary2]):
        with open(os.path.join(output_dir, f"{label}_avg_stats.txt"), 'w') as f:
            f.write(f"Metric\tMean\tMedian\tStandard Error\n")
            for key in stats_summary:
                mean, median, stderr = stats_summary[key]
                f.write(f"{key}\t{mean:.6f}\t{median:.6f}\t{stderr:.6f}\n")

    # Save p-values to a text file with 6 significant decimals
    with open(os.path.join(output_dir, filename.replace('.png', '_pvalues.txt')), 'w') as f:
        for key, p_value, label in zip(valid_metrics, p_values, significance_labels):
            f.write(f'{key}: p={p_value:.6f}, {label}\n')

'''
def plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2, metric_keys, filename, data_labels=None):
    print("\nCreating violin plots for", filename)
    valid_metrics = []
    p_values = []
    significance_labels = []

    # Set default data labels if none provided
    if data_labels is None:
        data_labels = ['Dataset 1', 'Dataset 2']
    else:
        data_labels = data_labels.split(',')

    # Determine valid metrics
    for key in metric_keys:
        data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])

        # Skip if data contains NaN values
        if np.isnan(data1).all() or np.isnan(data2).all():
            print(f"Skipping {key} due to invalid data")
            continue

        # Check normality and select the appropriate test
        normal1 = is_normal(data1)
        normal2 = is_normal(data2)

        if normal1 and normal2:
            stat, p_value = ttest_ind(data1, data2)
        else:
            stat, p_value = mannwhitneyu(data1, data2)

        # Skip if p-value is NaN
        if np.isnan(p_value):
            print(f"Skipping {key} due to NaN p-value")
            continue

        valid_metrics.append(key)
        p_values.append(p_value)

        # Determine significance label
        if p_value < 0.001:
            significance_labels.append("***")
        elif p_value < 0.01:
            significance_labels.append("**")
        elif p_value < 0.05:
            significance_labels.append("*")
        else:
            significance_labels.append("ns")

    num_metrics = len(valid_metrics)

    # If no valid metrics, print a warning and exit the function
    if num_metrics == 0:
        print("No valid metrics to plot.")
        return

    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 8))  # Increase height for better layout

    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Color palette for distinction

    # Plot valid metrics
    for i, key in enumerate(valid_metrics):
        ax = axes[i] if num_metrics > 1 else axes  # Handle case of single subplot
        data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])

        # Adjusting y-limits to start from 0 or the minimum value PLUS PADDING
        min_val1 = np.min(data1)
        max_val1 = np.max(data1)
        min_val2 = np.min(data2)
        max_val2 = np.max(data2)

        global_min = min(min_val1, min_val2)
        global_max = max(max_val1, max_val2)

        datarange = global_max - global_min
        global_min_padded = global_min - datarange * 0.1
        global_max_padded = global_max + datarange * 0.1
        ax.set_ylim(global_min_padded, global_max_padded)

        # Plot violin plot for Dataset 1
        parts1 = ax.violinplot(data1, positions=[1], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts1['bodies']:
            pc.set_facecolor(color_palette[i % len(color_palette)])
            pc.set_alpha(0.3)  # Set transparency for violin plot overlay
            pc.set_edgecolor('black')

        # Overlay scatter plot of data points for Dataset 1
        ax.scatter(np.full_like(data1, 1), data1, color=color_palette[i % len(color_palette)], alpha=0.7, edgecolor='k')

        # Custom additions for quartiles, medians, and extrema for Dataset 1
        quartile1, med1, quartile3 = np.percentile(data1, [25, 50, 75])
        mean1 = np.mean(data1)

        ax.scatter(1, med1, color='red', zorder=3)  # Median (Red)
        ax.scatter(1, mean1, color='white', zorder=3)  # Mean (White)
        ax.vlines(1, min_val1, max_val1, color='black', linestyle='-', lw=2)  # Min-Max range (Black)
        ax.vlines(1, quartile1, quartile3, color='black', linestyle='-', lw=5)  # Interquartile range (Black)

        # Label the distribution as Gaussian or Non-Gaussian for Dataset 1
        label1 = 'Gaussian' if normal1 else 'Non-Gaussian'
        ax.text(1, global_max_padded, label1, ha='center', va='bottom', fontsize=10, color='black')

        # Plot violin plot for Dataset 2
        parts2 = ax.violinplot(data2, positions=[2], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts2['bodies']:
            pc.set_facecolor(color_palette[i % len(color_palette)])
            pc.set_alpha(0.3)  # Set transparency for violin plot overlay
            pc.set_edgecolor('black')

        # Overlay scatter plot of data points for Dataset 2
        ax.scatter(np.full_like(data2, 2), data2, color=color_palette[i % len(color_palette)], alpha=0.7, edgecolor='k')

        # Custom additions for quartiles, medians, and extrema for Dataset 2
        quartile1, med2, quartile3 = np.percentile(data2, [25, 50, 75])
        mean2 = np.mean(data2)

        ax.scatter(2, med2, color='red', zorder=3)  # Median (Red)
        ax.scatter(2, mean2, color='white', zorder=3)  # Mean (White)
        ax.vlines(2, min_val2, max_val2, color='black', linestyle='-', lw=2)  # Min-Max range (Black)
        ax.vlines(2, quartile1, quartile3, color='black', linestyle='-', lw=5)  # Interquartile range (Black)

        # Label the distribution as Gaussian or Non-Gaussian for Dataset 2
        label2 = 'Gaussian' if normal2 else 'Non-Gaussian'
        ax.text(2, global_max_padded, label2, ha='center', va='bottom', fontsize=10, color='black')

        # Display numerical p-value and significance level, along with N for each dataset
        N1, N2 = len(data1), len(data2)
        ax.set_title(f'{key} (p={p_values[i]:.4f}, {significance_labels[i]})\nN={N1}, N={N2}', fontsize=12, pad=30)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(data_labels)

        # Adjusting slice image positions
        offset = 0.15  # Offset to ensure no overlap
        ax_min1 = fig.add_axes([0.05 + i * (1/num_metrics), -0.1, 0.1, 0.1])
        ax_min1.imshow(extreme_slices1[key]['min_slice'], cmap='gray')
        ax_min1.axis('off')

        ax_max1 = fig.add_axes([0.05 + i * (1/num_metrics), 1.02, 0.1, 0.1])
        ax_max1.imshow(extreme_slices1[key]['max_slice'], cmap='gray')
        ax_max1.axis('off')

        ax_min2 = fig.add_axes([0.15 + i * (1/num_metrics), -0.1, 0.1, 0.1])
        ax_min2.imshow(extreme_slices2[key]['min_slice'], cmap='gray')
        ax_min2.axis('off')

        ax_max2 = fig.add_axes([0.15 + i * (1/num_metrics), 1.02, 0.1, 0.1])
        ax_max2.imshow(extreme_slices2[key]['max_slice'], cmap='gray')
        ax_max2.axis('off')

    # Adjust the subplot layout to prevent overlap and ensure correct positioning
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Saved {filename} successfully.")

    # Save p-values to a text file
    with open(os.path.join(output_dir, filename.replace('.png', '_pvalues.txt')), 'w') as f:
        for key, p_value, label in zip(valid_metrics, p_values, significance_labels):
            f.write(f'{key}: p={p_value:.4f}, {label}\n')

'''

def save_raw_data(metrics1, metrics2, output_dir):
    """Save raw data to text files for each metric."""
    for metric in metrics1:
        np.savetxt(os.path.join(output_dir, f'{metric}_dataset1.txt'), np.array(metrics1[metric]), fmt='%s')
        np.savetxt(os.path.join(output_dir, f'{metric}_dataset2.txt'), np.array(metrics2[metric]), fmt='%s')

def plot_metrics(metrics, output_dir):
    """Plot metrics in grouped histograms."""
    keys = ['contrast', 'homogeneity', 'energy', 'correlation', 'kurtosis', 'entropy', 'mean', 'std_dev']
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))  # Adjust layout to fit all keys on two rows
    for i, key in enumerate(keys):
        ax = axes[i // 4, i % 4]
        data = np.array(metrics[key])
        if data.size > 0:
            ax.hist(data, bins='auto', alpha=0.7)
            ax.set_title(f'{key} Distribution')
            ax.set_xlabel(key)
            ax.set_ylabel('Frequency')
            np.savetxt(os.path.join(output_dir, f'{key}_data.txt'), data)  # Save histogram data
        else:
            ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_distributions.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Process image files to compute metrics.")
    parser.add_argument('--input', required=True, help="Comma-separated file paths for one or two image files.")
    parser.add_argument('--output_dir', default="molstats", help="Directory to save the outputs.")
    parser.add_argument('--data_labels', default=None, help="Comma-separated labels for the datasets.")
    args = parser.parse_args()

    input_files = args.input.split(',')
    output_dir = create_output_directory(args.output_dir)
    logger = setup_logger(output_dir)

    try:
        logger.info("Program started")
        start_time = time.time()
        if len(input_files) == 2:
            # Load and process data for the first file
            image_3d1, metrics1, extreme_slices1 = load_and_process_data(input_files[0])
            # Load and process data for the second file
            image_3d2, metrics2, extreme_slices2 = load_and_process_data(input_files[1])

            #save_raw_data(metrics1, output_dir)  # Saving raw data after metrics calculation
            #save_raw_data(metrics2, output_dir)

            save_raw_data(metrics1, metrics2, output_dir)

            haralick_keys = ['contrast', 'correlation', 'energy', 'homogeneity']
            other_metrics_keys = ['kurtosis', 'entropy', 'mean', 'std_dev']

            # Plotting Haralick features
            plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2, haralick_keys, "haralick_features.png", data_labels=args.data_labels)

            # Plotting other metric
            plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2, other_metrics_keys, "other_metrics.png", data_labels=args.data_labels)
        elif len(input_files) == 1:
            # Load and process data for the single file
            image_3d, metrics, extreme_slices = load_and_process_data(input_files[0])
            # Plot metrics
            plot_metrics(metrics, output_dir, extreme_slices)
        else:
            logger.error("Incorrect number of input files. Please provide one or two image files.")
    except Exception as e:
        logger.error(f"Program failed with error: {e}")
        logger.error(traceback.format_exc())
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)


if __name__ == '__main__':
    main()
