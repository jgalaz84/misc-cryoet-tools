#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 08/2019; last modification: jun/2024

import h5py
import argparse
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import kurtosis, ttest_ind
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sys import argv
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

def load_image_data(file_path):
    """ Load 3D image data from a file path, checking for the correct dataset. """
    print(f"Attempting to load data from {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            paths = list_hdf5_paths(file_path)
            print(f"Available dataset paths in {file_path}: {paths}")
            image_dataset_path = next((path for path in paths if path.endswith('/image')), None)
            if image_dataset_path:
                return np.array(file[image_dataset_path])
            else:
                raise KeyError("The dataset path containing '/image' does not exist in the file.")
    elif file_extension == '.mrc':
        with mrcfile.open(file_path, permissive=True) as mrc:
            return mrc.data
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")



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
    print('\nAnalyzing slices...')
    extremes = {key: {'min': (None, float('inf')), 'max': (None, float('-inf'))} for key in metrics}

    for idx, slice in enumerate(image_3d):
        mean = np.mean(slice)
        sigma = np.std(slice)
        if sigma > 0:
            slice_normalized = (slice * 255).astype(np.uint8) if slice.dtype in [np.float32, np.float64] else slice
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
            for key in metrics:
                metrics[key].append(None)  # Append None or a default value if the slice is not processed

    # Extract slices for extreme values
    extreme_slices = {key: {'min_slice': image_3d[extremes[key]['min'][0]], 'max_slice': image_3d[extremes[key]['max'][0]]} for key in metrics if extremes[key]['min'][0] is not None and extremes[key]['max'][0] is not None}
    return extreme_slices

def plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2, metric_keys, filename):
    print("\nCreating violin plots for", filename)
    num_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 8))  # Increase height for better layout

    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Color palette for distinction

    for i, key in enumerate(metric_keys):
        ax = axes[i]
        data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])
        
        # Plot violin plot for Dataset 1
        parts1 = ax.violinplot(data1, positions=[1], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts1['bodies']:
            pc.set_facecolor(color_palette[i % len(color_palette)])
            pc.set_alpha(1)
            pc.set_edgecolor('black')

        # Custom additions for quartiles, medians, and extrema for Dataset 1
        quartile1, med1, quartile3 = np.percentile(data1, [25, 50, 75])
        min_val1 = np.min(data1)
        max_val1 = np.max(data1)
        mean1 = np.mean(data1)

        ax.scatter(1, med1, color='red', zorder=3)  # Median (Red)
        ax.scatter(1, mean1, color='white', zorder=3)  # Mean (White)
        ax.vlines(1, min_val1, max_val1, color='black', linestyle='-', lw=2)  # Min-Max range (Black)
        ax.vlines(1, quartile1, quartile3, color='black', linestyle='-', lw=5)  # Interquartile range (Black)

        # Plot violin plot for Dataset 2
        parts2 = ax.violinplot(data2, positions=[2], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts2['bodies']:
            pc.set_facecolor(color_palette[i % len(color_palette)])
            pc.set_alpha(0.5)
            pc.set_edgecolor('black')

        # Custom additions for quartiles, medians, and extrema for Dataset 2
        quartile1, med2, quartile3 = np.percentile(data2, [25, 50, 75])
        min_val2 = np.min(data2)
        max_val2 = np.max(data2)
        mean2 = np.mean(data2)

        ax.scatter(2, med2, color='red', zorder=3)  # Median (Red)
        ax.scatter(2, mean2, color='white', zorder=3)  # Mean (White)
        ax.vlines(2, min_val2, max_val2, color='black', linestyle='-', lw=2)  # Min-Max range (Black)
        ax.vlines(2, quartile1, quartile3, color='black', linestyle='-', lw=5)  # Interquartile range (Black)

        ax.set_title(f'{key} (p={ttest_ind(data1, data2).pvalue:.4f})')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Dataset 1', 'Dataset 2'])

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
    plt.savefig(os.path.join(output_dir, filename),bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Saved {filename} successfully.")

'''
kinda WORKS
def plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2, metric_keys, filename):
    print("\nCreating violin plots for", filename)
    num_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 6))  # Adjust size as needed

    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Fewer colors, cycling through

    for i, key in enumerate(metric_keys):
        ax = axes[i]
        data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])
        
        parts1 = ax.violinplot(data1, positions=[1], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts1['bodies']:
            pc.set_facecolor(color_palette[i % len(color_palette)])
            pc.set_alpha(1)

        parts2 = ax.violinplot(data2, positions=[2], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts2['bodies']:
            pc.set_facecolor(color_palette[i % len(color_palette)])
            pc.set_alpha(0.5)

        # Adding mean and median indicators
        ax.scatter(1, np.mean(data1), color='red', s=30, zorder=3)  # Red star for mean
        ax.scatter(1, np.median(data1), color='white', s=30, edgecolor='black', zorder=3)  # White circle for median
        ax.scatter(2, np.mean(data2), color='red', s=30, zorder=3)
        ax.scatter(2, np.median(data2), color='white', s=30, edgecolor='black', zorder=3)

        ax.set_title(f'{key} (p={ttest_ind(data1, data2).pvalue:.3e})')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Dataset 1', 'Dataset 2'])

        # Add extreme slice images correctly aligned with the plots
        ax_min1 = fig.add_axes([0.07 + i * 0.25, 0.02, 0.1, 0.1])
        ax_min1.imshow(extreme_slices1[key]['min_slice'], cmap='gray')
        ax_min1.axis('off')

        ax_max1 = fig.add_axes([0.07 + i * 0.25, 0.88, 0.1, 0.1])
        ax_max1.imshow(extreme_slices1[key]['max_slice'], cmap='gray')
        ax_max1.axis('off')

        ax_min2 = fig.add_axes([0.17 + i * 0.25, 0.02, 0.1, 0.1])
        ax_min2.imshow(extreme_slices2[key]['min_slice'], cmap='gray')
        ax_min2.axis('off')

        ax_max2 = fig.add_axes([0.17 + i * 0.25, 0.88, 0.1, 0.1])
        ax_max2.imshow(extreme_slices2[key]['max_slice'], cmap='gray')
        ax_max2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"Saved {filename} successfully.")
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
            plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2, haralick_keys, "haralick_features.png")

            # Plotting other metrics
            plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2, other_metrics_keys, "other_metrics.png")
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


def load_and_process_data(file_path):
    """ Helper function to load image data and calculate metrics. """
    image_3d = load_image_data(file_path)
    if image_3d is not None:
        metrics, extreme_slices = calculate_metrics_for_images(image_3d)
        return image_3d, metrics, extreme_slices
    else:
        raise Exception(f"Failed to load image data from {file_path}")

if __name__ == '__main__':
    main()



'''
def main():
    parser = argparse.ArgumentParser(description="Process image files to compute metrics.")
    parser.add_argument('--input', required=True, help="Comma-separated file paths for one or two image files.")
    parser.add_argument('--output_dir', default="molstats", help="Directory to save the outputs.")
    args = parser.parse_args()

    input_files = args.input.split(',')
    output_dir = create_output_directory(args.output_dir)
    logger = setup_logger(output_dir)

    try:
        logger.info("Program started")
        start_time = time.time()
        if len(input_files) == 2:
            image_3d1 = load_image_data(input_files[0])
            image_3d2 = load_image_data(input_files[1])
            metrics1 = calculate_metrics_for_images(input_files[0])
            extreme_slices1 = process_slices(image_3d1, metrics1)
            metrics2 = calculate_metrics_for_images(input_files[1])
            extreme_slices2 = process_slices(image_3d2, metrics2)
            plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices1, extreme_slices2)
        elif len(input_files) == 1:
            image_3d = load_image_data(input_files[0])
            metrics = calculate_metrics_for_images(input_files[0])
            extreme_slices = process_slices(image_3d, metrics)
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



          '''




'''
def plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices):
    """Plot comparative violin plots for two sets of metrics with extreme slices displayed."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 24))  # Increased height to accommodate slice images
    keys = sorted(metrics1.keys())
    significance_markers = ["ns", "*", "**", "***", "****"]
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, key in enumerate(keys):
        ax = axes[i // 4, i % 4]
        data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])
        # Determine indices for extreme values
        min_idx1, max_idx1 = np.argmin(data1), np.argmax(data1)
        min_idx2, max_idx2 = np.argmin(data2), np.argmax(data2)
        
        # Plot violin
        parts = ax.violinplot([data1, data2], positions=[1, 2], showmeans=True, showmedians=True, showextrema=True)
        for pc, color in zip(parts['bodies'], [color_palette[i], color_palette[i]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        
        # Set colors for other parts of the violin plot
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            vp = parts.get(partname)
            vp.set_edgecolor(color_palette[i])
            vp.set_linewidth(2)
        
        ax.set_title(f'{key} (p={np.format_float_scientific(ttest_ind(data1, data2).pvalue, precision=3)} {significance_markers[np.random.randint(0, 5)]})')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Dataset 1', 'Dataset 2'])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Show extreme slices above and below the plot
        ax_extreme_min = fig.add_axes([0.91, 0.76 - (i * 0.08), 0.07, 0.07])  # Adjust these positions as needed
        ax_extreme_min.imshow(extreme_slices[key][min_idx1])  # Assuming images are appropriately scaled
        ax_extreme_min.axis('off')
        
        ax_extreme_max = fig.add_axes([0.91, 0.68 - (i * 0.08), 0.07, 0.07])
        ax_extreme_max.imshow(extreme_slices[key][max_idx1])
        ax_extreme_max.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_violin_plots.png"))
    plt.close(fig)
'''




'''
def plot_comparative_violin_plots(metrics1, metrics2, output_dir, extreme_slices):
    """Plot comparative violin plots for two sets of metrics with extreme slices displayed."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 24))  # Increased height to accommodate slice images
    keys = sorted(metrics1.keys())
    significance_markers = ["ns", "*", "**", "***", "****"]
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, key in enumerate(keys):
        ax = axes[i // 4, i % 4]
        data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])
        
        # Plot violin
        parts = ax.violinplot([data1, data2], positions=[1, 2], showmeans=True, showmedians=True, showextrema=True)
        for pc, color in zip(parts['bodies'], [color_palette[i], color_palette[i]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        
        # Set colors for lines to black
        for partname in ('cmeans', 'cmedians', 'cmaxes', 'cmins', 'cbars'):
            vp = parts.get(partname)
            vp.set_edgecolor('black')
            vp.set_linewidth(2)
        
        ax.set_title(f'{key} (p={np.format_float_scientific(ttest_ind(data1, data2).pvalue, precision=3)} {significance_markers[np.random.randint(0, 5)]})')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Dataset 1', 'Dataset 2'])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Show extreme slices above and below the plot
        ax_extreme_min = fig.add_axes([0.91, 0.76 - (i * 0.08), 0.07, 0.07])  # Adjust these positions as needed
        ax_extreme_min.imshow(extreme_slices[key]['min_slice'], cmap='gray')  # Ensure the correct key is used here
        ax_extreme_min.axis('off')
        
        ax_extreme_max = fig.add_axes([0.91, 0.68 - (i * 0.08), 0.07, 0.07])
        ax_extreme_max.imshow(extreme_slices[key]['max_slice'], cmap='gray')
        ax_extreme_max.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_violin_plots.png"))
    plt.close(fig)
    '''












'''
def process_slices(image_3d, metrics):
    print("\nAnalyzing slices...")  # Print statement for debugging
    """Process each slice and calculate metrics."""
    for slice in image_3d:
        mean = np.mean(slice)
        sigma = np.std(slice)
        if sigma > 0:
            slice = (slice * 255).astype(np.uint8) if slice.dtype in [np.float32, np.float64] else slice
            glcm = graycomatrix(slice, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
            metrics['kurtosis'].append(kurtosis(slice.ravel(), fisher=True))
            metrics['entropy'].append(shannon_entropy(slice))
            metrics['contrast'].append(graycoprops(glcm, 'contrast')[0].mean())
            metrics['homogeneity'].append(graycoprops(glcm, 'homogeneity')[0].mean())
            metrics['energy'].append(graycoprops(glcm, 'energy')[0].mean())
            metrics['correlation'].append(graycoprops(glcm, 'correlation')[0].mean())
        metrics['mean'].append(mean)
        metrics['std_dev'].append(sigma)

def plot_comparative_violin_plots(metrics1, metrics2, output_dir):
    """Plot comparative violin plots for two sets of metrics, separated by type."""
    haralick_keys = ['contrast', 'homogeneity', 'energy', 'correlation']
    other_keys = ['kurtosis', 'entropy', 'mean', 'std_dev']
    metrics_groups = [haralick_keys, other_keys]
    colors = ['blue', 'green', 'red', 'purple']

    for keys, filename in zip(metrics_groups, ["haralick_features.png", "other_metrics.png"]):
        fig, axes = plt.subplots(1, len(keys), figsize=(12, 6))
        for i, key in enumerate(keys):
            ax = axes[i]
            data1, data2 = np.array(metrics1[key]), np.array(metrics2[key])
            save_raw_data(data1, data2, key, output_dir)  # Ensure raw data is saved

            if len(data1) > 0 and len(data2) > 0:
                # Statistical test
                p_value = ttest_ind(data1, data2).pvalue
                significance = '****' if p_value < 0.0001 else '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                
                # Plotting
                parts = ax.violinplot([data1, data2], showmeans=True, showmedians=True, showextrema=True, positions=[1, 2])
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
                
                # Adding mean, median, and interquartile range
                quartile1, medians, quartile3 = np.percentile(data1, [25, 50, 75])
                whisker1, whisker2 = np.percentile(data2, [25, 75])
                ax.scatter([1], [np.mean(data1)], color='white', zorder=3)
                ax.scatter([2], [np.mean(data2)], color='white', zorder=3)
                ax.vlines([1], quartile1, quartile3, color='k', linestyle='-', lw=5)
                ax.vlines([2], whisker1, whisker2, color='k', linestyle='-', lw=5)
                ax.set_title(f'{key} (p={p_value:.3f} {significance})')
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Dataset 1', 'Dataset 2'])
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            else:
                ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
'''

'''
def main():
    parser = argparse.ArgumentParser(description="Process image files to compute metrics.")
    parser.add_argument('--input', required=True, help="Comma-separated file paths for one or two image files.")
    parser.add_argument('--output_dir', default="molstats", help="Directory to save the outputs.")
    args = parser.parse_args()

    input_files = args.input.split(',')
    output_dir = create_output_directory(args.output_dir)
    logger = setup_logger(output_dir)

    try:
        logger.info("Program started")
        start_time = time.time()
        if len(input_files) == 2:
            metrics1 = calculate_metrics_for_images(input_files[0])
            metrics2 = calculate_metrics_for_images(input_files[1])
            plot_comparative_violin_plots(metrics1, metrics2, output_dir)  # Ensure this function is correctly defined to plot and save figures.
        elif len(input_files) == 1:
            metrics = calculate_metrics_for_images(input_files[0])
            plot_metrics(metrics, output_dir)  # Ensure this function is correctly defined to plot and save figures.
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
'''
