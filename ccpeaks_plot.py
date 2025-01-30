#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2025; last modification: Jan/30/2025

import argparse
import os
import numpy as np
import mrcfile
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, normaltest
import logging
import time
import traceback
from mpl_toolkits.mplot3d import Axes3D

def create_output_directory(base_dir):
    """Create and return a numbered output directory."""
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "correlation_files"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "coordinate_files"), exist_ok=True)
    return output_dir

def load_image(file_path):
    """Loads a 3D image or a stack of 3D images from an HDF5 or MRC file."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            dataset_paths = [f"MDF/images/{i}/image" for i in range(len(file["MDF/images"]))]
            image_stack = np.array([file[ds][:] for ds in dataset_paths])
    elif ext == '.mrc':
        with mrcfile.open(file_path, permissive=True) as mrc:
            image_stack = mrc.data
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return image_stack

def compute_ncc_map_3d(image_stack, template):
    """Computes the 3D normalized cross-correlation map for each image in the stack."""
    ncc_maps = []
    i=0
    for image in image_stack:
        print(f"\nfor image {i}, Image shape {image.shape}, Template shape {template.shape}")
        if image.shape != template.shape:
            raise ValueError(f"Mismatch in dimensions: Image shape {image.shape}, Template shape {template.shape}")
        
        ncc = correlate(image, template, mode='same', method='fft')
        ncc = (ncc - np.mean(ncc)) / np.std(ncc)
        ncc_maps.append(ncc)
        i+=1

    return np.array(ncc_maps)

def extract_ncc_peaks(ncc_map, npeaks):
    """Extracts the top NCC peaks and their coordinates from the 3D NCC map."""
    flat_indices = np.argsort(ncc_map.ravel())[::-1][:npeaks]
    peak_values = ncc_map.ravel()[flat_indices]
    peak_coords = np.column_stack(np.unravel_index(flat_indices, ncc_map.shape))
    return peak_values, peak_coords

def save_peak_data(output_dir, peak_coords, peak_values, filename, index):
    """Saves the peak coordinates and values to a formatted text file."""
    output_txt = os.path.join(output_dir, 'correlation_files', f'ccc_{filename}_' + str(index).zfill(3)+ '.txt')
    np.savetxt(output_txt, np.column_stack((peak_coords, peak_values)), fmt='%5d %5d %5d %10.6f', header='X Y Z NCC_Value')

def plot_3d_peak_coordinates(output_dir, peak_coords, filename, index):
    """Plots the NCC peak coordinates in 3D."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2], c='red', marker='o')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Top NCC Peak Locations")
    output_png = os.path.join(output_dir, 'coordinate_files', f'ccc_{filename}_' + str(index).zfill(3)+ '.png')
    plt.savefig(output_png)
    plt.close()

def is_normal(data, alpha=0.05):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    _, p_value_shapiro = shapiro(data)
    return p_value_shapiro > alpha

def calculate_effect_size(data1, data2):
    """Calculates Cohen's d or Rank-Biserial correlation depending on normality."""
    normal1, normal2 = is_normal(data1), is_normal(data2)
    if normal1 and normal2:
        stat, p_value = ttest_ind(data1, data2)
        effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
        method = "Cohen's d"
    else:
        stat, p_value = mannwhitneyu(data1, data2)
        effect_size = 1 - (2 * stat) / (len(data1) * len(data2))
        method = "Rank-biserial correlation"
    return p_value, effect_size, method


def plot_violin(data1, data2, filenames, output_dir):
    """Generates a violin plot comparing NCC peak distributions with relevant statistics, including number of peaks (N)."""
    
    p_value, effect_size, method = calculate_effect_size(data1, data2)
    normal1, normal2 = is_normal(data1), is_normal(data2)

    fig, ax = plt.subplots(figsize=(6, 8))
    parts = ax.violinplot([data1, data2], showmeans=True, showmedians=True)

    # Customizing mean and median appearance
   
    if 'cmeans' in parts:
        parts['cmeans'].set_linestyle('--')  # Dashed style for means
        parts['cmeans'].set_linewidth(2)
        parts['cmeans'].set_color('red')
    
    if 'cmedians' in parts:
        parts['cmedians'].set_linestyle('-')  # Solid style for medians
        parts['cmedians'].set_linewidth(2)
        parts['cmedians'].set_color('black')

    #if 'cmeans' in parts:
    #    for line in parts['cmeans']:
    #        line.set_linestyle('--')  # Dashed style for means
    #        line.set_linewidth(2)
    #        line.set_color('red')
    
    #if 'cmedians' in parts:
    #    for line in parts['cmedians']:
    #        line.set_linestyle('-')  # Solid style for medians
    #        line.set_linewidth(2)
    #        line.set_color('black')
    

    colors = ['blue', 'orange']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(filenames)
    ax.set_ylabel("NCC Peak Values")
    ax.set_title(f"Comparison of NCC Peaks\nMethod: {method}, p={p_value:.6f}, Effect Size: {effect_size:.6f}")

    # Adjust text placement to prevent overlap
    text_y_offset = max(max(data1), max(data2)) * 0.05
    for i, data in enumerate([data1, data2]):
        mean_val, median_val = np.mean(data), np.median(data)
        ax.text(i + 1, max(data) + text_y_offset, 
                f"N={len(data)}\nGaussian: {is_normal(data)}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}",
                ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ncc_violin_plot.png"))
    plt.close()
    
'''
def plot_violin(data1, data2, filenames, output_dir):
    """Generates a violin plot comparing NCC peak distributions with relevant statistics, including number of peaks (N)."""
    
    p_value, effect_size, method = calculate_effect_size(data1, data2)
    normal1, normal2 = is_normal(data1), is_normal(data2)

    fig, ax = plt.subplots(figsize=(6, 8))
    parts = ax.violinplot([data1, data2], showmeans=True, showmedians=True)

    parts = ax.violinplot([data1, data2], showmeans=True, showmedians=True)

    # Customizing mean and median appearance
    for line in parts['cmeans']:  # Mean lines
        line.set_linestyle('--')  # Dashed style for means
        line.set_linewidth(2)
        line.set_color('red')

    for line in parts['cmedians']:  # Median lines
        line.set_linestyle('-')  # Solid style for medians
        line.set_linewidth(2)
        line.set_color('black')

    colors = ['blue', 'orange']
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(filenames)
    ax.set_ylabel("NCC Peak Values")
    ax.set_title(f"Comparison of NCC Peaks\nMethod: {method}, p={p_value:.6f}, Effect Size: {effect_size:.6f}")

    # Add text with the number of peaks (N), Gaussianity, mean, and median above each violin plot

    
    #max_y = max(max(data1), max(data2))
    #y_offset = (max_y - min(min(data1), min(data2))) * 0.1  # Adjust text positioning
    #
    #for i, data in enumerate([data1, data2]):
    #    mean_val, median_val = np.mean(data), np.median(data)
    #    ax.text(i + 1, max_y + y_offset, 
    #            f"N={len(data)}\nGaussian: {normal1 if i == 0 else normal2}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}",
    #            ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    #            

    # Adjust text placement to prevent overlap
    text_y_offset = max(max(data1), max(data2)) * 0.05
    for i, data in enumerate([data1, data2]):
        mean_val, median_val = np.mean(data), np.median(data)
        ax.text(i + 1, max(data) + text_y_offset, 
                f"N={len(data)}\nGaussian: {is_normal(data)}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}",
                ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ncc_violin_plot.png"))
    plt.close()
    '''


def main():
    parser = argparse.ArgumentParser(description="Compute 3D NCC maps and compare peak distributions.")
    parser.add_argument('--input', required=True, help="Comma-separated input file paths.")
    parser.add_argument('--template', required=True, help="Template image file path.")
    parser.add_argument('--npeaks', type=int, default=10, help="Number of top peaks to extract per image in a stack.")
    parser.add_argument('--output_dir', default="ncc_analysis", help="Output directory base name.")
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    input_files = args.input.split(',')
    
    #template = load_image(args.template)
    template = load_image(args.template).squeeze()
    
    peak_values_list = []
    filenames = []
    
    index=0
    for file_path in input_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        image_stack = load_image(file_path)

        # Compute NCC for each image in the stack
        ncc_maps = compute_ncc_map_3d(image_stack, template)
        
        all_peak_values = []
        all_peak_coords = []
        
        for ncc_map in ncc_maps:
            peak_values, peak_coords = extract_ncc_peaks(ncc_map, args.npeaks)
            all_peak_values.extend(peak_values)
            all_peak_coords.extend(peak_coords)

            save_peak_data(output_dir, peak_coords, peak_values, filename, index)
            plot_3d_peak_coordinates(output_dir, peak_coords, filename, index)

            index+=1

        peak_values_list.append(all_peak_values)
        filenames.append(filename)

    if len(peak_values_list) == 2:
        plot_violin(*peak_values_list, filenames, output_dir)

if __name__ == '__main__':
    main()


'''
SINGLE IMAGE PROCESSING PER INPUT

#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2025; last modification: Jan/30/2025

import argparse
import os
import numpy as np
import mrcfile
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, normaltest
import logging
import time
import traceback
from mpl_toolkits.mplot3d import Axes3D

def create_output_directory(base_dir):
    """Create and return a numbered output directory."""
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    return output_dir

def load_image(file_path):
    """Loads a 3D image from an HDF5 or MRC file."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            dataset_path = "MDF/images/0/image"
            if dataset_path not in file:
                raise KeyError(f"Dataset path '{dataset_path}' not found in HDF5 file.")
            print(f"Using dataset: {dataset_path}")
            image_stack = file[dataset_path][:]
    elif ext == '.mrc':
        with mrcfile.open(file_path, permissive=True) as mrc:
            image_stack = mrc.data
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return image_stack

def compute_ncc_map_3d(image, template):
    """Computes the 3D normalized cross-correlation map without redundant normalization."""
    
    # Normalize inputs
    #image = (image - np.mean(image)) / np.std(image)
    #template = (template - np.mean(template)) / np.std(template)
    
    # Compute NCC
    ncc = correlate(image, template, mode='same', method='fft')
    ncc = (ncc - np.mean(ncc)) / np.std(ncc)
    
    return ncc

def extract_ncc_peaks(ncc_map, npeaks):
    """Extracts the top NCC peaks and their coordinates from the 3D NCC map."""
    flat_indices = np.argsort(ncc_map.ravel())[::-1][:npeaks]
    peak_values = ncc_map.ravel()[flat_indices]
    peak_coords = np.column_stack(np.unravel_index(flat_indices, ncc_map.shape))
    return peak_values, peak_coords

def save_peak_data(output_dir, filename, peak_coords, peak_values):
    """Saves the peak coordinates and values to a formatted text file."""
    output_txt = os.path.join(output_dir, f'ccc_{filename}.txt')
    np.savetxt(output_txt, np.column_stack((peak_coords, peak_values)), fmt='%5d %5d %5d %10.6f', header='X Y Z NCC_Value')

def plot_3d_peak_coordinates(peak_coords, filename):
    """Plots the NCC peak coordinates in 3D."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2], c='red', marker='o')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Top NCC Peak Locations")
    plt.savefig(filename)
    plt.close()

def plot_violin(data1, data2, filenames, output_dir, tag=''):
    """Generates a violin plot comparing NCC peak distributions with relevant statistics."""
    
    p_value, effect_size, method = calculate_effect_size(data1, data2)
    normal1, normal2 = is_normal(data1), is_normal(data2)
    
    fig, ax = plt.subplots(figsize=(6, 8))
    parts = ax.violinplot([data1, data2], showmeans=True, showmedians=True)
    colors = ['blue', 'orange']
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(filenames)
    ax.set_ylabel("NCC Peak Values")
    ax.set_title(f"Comparison of NCC Peaks\nMethod: {method}, p={p_value:.6f}, Effect Size: {effect_size:.6f}")
    
    max_y = max(max(data1), max(data2))
    y_offset = (max_y - min(min(data1), min(data2))) * 0.15  
    
    for i, data in enumerate([data1, data2]):
        mean_val, median_val = np.mean(data), np.median(data)
        ax.text(i + 1, max_y + y_offset, 
                f"N={len(data)}\nGaussian: {normal1 if i == 0 else normal2}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}",
                ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    vp_filename = "ncc_violin_plot.png" if tag == '' else "ncc_violin_plot_norm.png"
    plt.savefig(os.path.join(output_dir, vp_filename))
    plt.close()

def calculate_effect_size(data1, data2):
    """Calculates Cohen's d or Rank-Biserial correlation depending on normality, with dataset normalization."""
    
    # Normalize both datasets (z-score normalization)
    #data1 = (data1 - np.mean(data1)) / np.std(data1)
    #data2 = (data2 - np.mean(data2)) / np.std(data2)
    
    normal1, normal2 = is_normal(data1), is_normal(data2)
    
    if normal1 and normal2:
        stat, p_value = ttest_ind(data1, data2)
        effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
        method = "Cohen's d"
    else:
        stat, p_value = mannwhitneyu(data1, data2)
        effect_size = 1 - (2 * stat) / (len(data1) * len(data2))
        method = "Rank-biserial correlation"
    
    return p_value, effect_size, method


def is_normal(data, alpha=0.05):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    _, p_value_shapiro = shapiro(data)
    #_, p_value_normal = normaltest(data)

    #print(f"\np_value_shapiro={p_value_shapiro})")
    #print(f"\np_value_normal={p_value_normal})")
    
    #return (p_value_shapiro+p_value_normal)/2 > alpha
    return p_value_shapiro > alpha

def main():
    parser = argparse.ArgumentParser(description="Compute 3D NCC maps and compare peak distributions.")
    parser.add_argument('--input', required=True, help="Comma-separated input file paths.")
    parser.add_argument('--template', required=True, help="Template image file path.")
    parser.add_argument('--npeaks', type=int, default=10, help="Number of top peaks to extract.")
    parser.add_argument('--output_dir', default="ncc_analysis", help="Output directory base name.")
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    input_files = args.input.split(',')
    
    template = load_image(args.template)
    
    peak_values_list = []
    filenames = []
    
    for file_path in input_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        image_stack = load_image(file_path)
        ncc_map = compute_ncc_map_3d(image_stack, template)
        
        peak_values, peak_coords = extract_ncc_peaks(ncc_map, args.npeaks)
        save_peak_data(output_dir, filename, peak_coords, peak_values)
        plot_3d_peak_coordinates(peak_coords, os.path.join(output_dir, f'ccc_{filename}_peaks.png'))
        peak_values_list.append(peak_values)
        filenames.append(filename)
    
    if len(peak_values_list) == 2:
        plot_violin(*peak_values_list, filenames, output_dir)

if __name__ == '__main__':
    main()

'''








'''
def plot_violin(data1, data2, filenames, output_dir, tag=''):
    """Generates a violin plot comparing NCC peak distributions with relevant statistics."""
    p_value, effect_size, method = calculate_effect_size(data1, data2)
    normal1, normal2 = is_normal(data1), is_normal(data2)
    
    plt.figure(figsize=(6, 8))
    parts = plt.violinplot([data1, data2], showmeans=True, showmedians=True)
    colors = ['blue', 'orange']
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
    
    plt.xticks([1, 2], filenames)
    plt.ylabel("NCC Peak Values")
    plt.title(f"Comparison of NCC Peaks\nMethod: {method}, p={p_value:.6f}, Effect Size: {effect_size:.6f}")
    
    for i, data in enumerate([data1, data2]):
        mean_val, median_val = np.mean(data), np.median(data)
        plt.text(i + 1, max(data) + (0.1 * max(data)), 
                 f"N={len(data)}\nGaussian: {normal1 if i == 0 else normal2}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}",
                 ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    vp_filename = "ncc_violin_plot.png" if tag == '' else "ncc_violin_plot_norm.png"
    plt.savefig(os.path.join(output_dir, vp_filename))
    plt.close()
'''



'''
def compute_ncc_map_3d(image, template):
    """Computes the 3D normalized cross-correlation map with correct normalization."""
    
    image = (image - np.mean(image)) / np.std(image)
    template = (template - np.mean(template)) / np.std(template)
    
    ncc = correlate(image, template, mode='same', method='fft')
    
    ncc_norm = (ncc - np.mean(ncc)) / np.std(ncc)

    return ncc, ncc_norm
'''


'''
def calculate_effect_size(data1, data2):
    """Calculates Cohen's d or Rank-Biserial correlation depending on normality."""
    normal1, normal2 = is_normal(data1), is_normal(data2)
    if normal1 and normal2:
        stat, p_value = ttest_ind(data1, data2)
        effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
        method = "Cohen's d"
    else:
        stat, p_value = mannwhitneyu(data1, data2)
        effect_size = 1 - (2 * stat) / (len(data1) * len(data2))
        method = "Rank-biserial correlation"
    return p_value, effect_size, method
    '''


'''
#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2025; last modification: Jan/29/2025

import argparse
import os
import numpy as np
import mrcfile
import h5py
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import logging
import time
import traceback
from mpl_toolkits.mplot3d import Axes3D

def create_output_directory(base_dir):
    """Create and return a numbered output directory."""
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    return output_dir

def load_image(file_path):
    """Loads a 3D image from an HDF5 or MRC file."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            dataset_path = "MDF/images/0/image"
            if dataset_path not in file:
                raise KeyError(f"Dataset path '{dataset_path}' not found in HDF5 file.")
            print(f"Using dataset: {dataset_path}")
            image_stack = file[dataset_path][:]
    elif ext == '.mrc':
        with mrcfile.open(file_path, permissive=True) as mrc:
            image_stack = mrc.data
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return image_stack

def compute_ncc_map_3d(image, template):
    """Computes the 3D normalized cross-correlation map with correct normalization."""
    
    image = (image - np.mean(image)) / np.std(image)
    template = (template - np.mean(template)) / np.std(template)
    
    ncc = correlate(image, template, mode='same', method='fft')
    
    #ncc /= (np.std(image) * np.std(template))
    ncc_norm = (ncc - np.mean(ncc)) / np.std(ncc)

    return ncc, ncc_norm

def extract_ncc_peaks(ncc_map, npeaks):
    """Extracts the top NCC peaks and their coordinates from the 3D NCC map."""
    flat_indices = np.argsort(ncc_map.ravel())[::-1][:npeaks]
    peak_values = ncc_map.ravel()[flat_indices]
    peak_coords = np.column_stack(np.unravel_index(flat_indices, ncc_map.shape))
    return peak_values, peak_coords

def save_peak_data(output_dir, filename, peak_coords, peak_values):
    """Saves the peak coordinates and values to a formatted text file."""
    output_txt = os.path.join(output_dir, f'ccc_{filename}.txt')
    np.savetxt(output_txt, np.column_stack((peak_coords, peak_values)), fmt='%5d %5d %5d %10.6f', header='X Y Z NCC_Value')

def plot_3d_peak_coordinates(peak_coords, filename):
    """Plots the NCC peak coordinates in 3D."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2], c='red', marker='o')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Top NCC Peak Locations")
    plt.savefig(filename)
    plt.close()

def plot_violin(data1, data2, filenames, output_dir, tag=''):
    """Generates a violin plot comparing NCC peak distributions with relevant statistics."""
    p_value, effect_size, method = calculate_effect_size(data1, data2)
    normal1, normal2 = is_normal(data1), is_normal(data2)
    
    plt.figure(figsize=(6, 8))
    parts = plt.violinplot([data1, data2], showmedians=True)
    colors = ['blue', 'orange']
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
    
    plt.xticks([1, 2], filenames)
    plt.ylabel("NCC Peak Values")
    plt.title(f"Comparison of NCC Peaks\nMethod: {method}, p={p_value:.6f}, Effect Size: {effect_size:.6f}")
    
    for i, data in enumerate([data1, data2]):
        plt.text(i + 1, max(data) + 0.02, f"N={len(data)}\nGaussian: {normal1 if i == 0 else normal2}", ha='center')
    
    plt.tight_layout()
    vp_filename = "ncc_violin_plot.png"
    print(f"\n\n\ntag={tag}")
    if tag == 'norm':
        vp_filename = "ncc_violin_plot_norm.png"
    plt.savefig(os.path.join(output_dir, vp_filename))
    plt.close()


def calculate_effect_size(data1, data2):
    """Calculates Cohen's d or Rank-Biserial correlation depending on normality."""
    normal1, normal2 = is_normal(data1), is_normal(data2)
    if normal1 and normal2:
        stat, p_value = ttest_ind(data1, data2)
        effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
        method = "Cohen's d"
    else:
        stat, p_value = mannwhitneyu(data1, data2)
        effect_size = 1 - (2 * stat) / (len(data1) * len(data2))
        method = "Rank-biserial correlation"
    return p_value, effect_size, method

def is_normal(data, alpha=0.05):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    _, p_value = shapiro(data)
    return p_value > alpha

def main():
    parser = argparse.ArgumentParser(description="Compute 3D NCC maps and compare peak distributions.")
    parser.add_argument('--input', required=True, help="Comma-separated input file paths.")
    parser.add_argument('--template', required=True, help="Template image file path.")
    parser.add_argument('--npeaks', type=int, default=10, help="Number of top peaks to extract.")
    parser.add_argument('--output_dir', default="ncc_analysis", help="Output directory base name.")
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    input_files = args.input.split(',')
    
    template = load_image(args.template)
    
    peak_values_list = []
    peak_values_norm_list = []
    filenames = []
    filenames_norm = []
    
    for file_path in input_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        image_stack = load_image(file_path)
        ncc_map, ncc_norm_map = compute_ncc_map_3d(image_stack, template)
        
        peak_values, peak_coords = extract_ncc_peaks(ncc_map, args.npeaks)
        save_peak_data(output_dir, filename, peak_coords, peak_values)
        
        peak_values_norm, peak_coords_norm = extract_ncc_peaks(ncc_norm_map, args.npeaks)
        filename_norm = filename + '_norm'
        save_peak_data(output_dir, filename_norm, peak_coords_norm, peak_values_norm)
        
        plot_3d_peak_coordinates(peak_coords, os.path.join(output_dir, f'ccc_{filename}_peaks.png'))
        peak_values_list.append(peak_values)
        peak_values_norm_list.append(peak_values_norm)
        filenames.append(filename)
        filenames_norm.append(filename_norm)
    
    if len(peak_values_list) == 2:
        plot_violin(*peak_values_list, filenames, output_dir)
    if len(peak_values_norm_list) == 2:
        print(f'\nplotting violin plot for filenames_norm={filenames_norm}')
        plot_violin(*peak_values_norm_list, filenames_norm, output_dir,'norm')

if __name__ == '__main__':
    main()
'''





'''
def compute_ncc_map_3d(image, template):
    """Computes the 3D normalized cross-correlation map."""
    image = (image - np.mean(image)) / np.std(image)
    template = (template - np.mean(template)) / np.std(template)
    
    ncc = correlate(image, template, mode='same', method='fft')
    
    # Normalize the entire NCC map to have mean 0 and standard deviation 1
    ncc = (ncc - np.mean(ncc)) / np.std(ncc)
    
    return ncc

'''

