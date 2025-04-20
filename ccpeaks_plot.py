#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2025
# Last modification: Apr/20/2025
# Performance optimizations: Apr/20/2025
#
# PERFORMANCE NOTES:
# - Added extensive memory optimizations to prevent crashes on large datasets
# - Improved FFT and correlation computation using vectorized operations
# - Optimized peak finding algorithms for better speed
# - Added auto-tuning of processing parameters based on available system resources
# - Added performance_mode parameter with options: 'memory', 'balanced', 'speed'
# - Default peak_method switched to 'vectorized' for better performance
# - Added robust process pool error recovery for BrokenProcessPool errors
# - Added memory monitoring and graceful degradation when memory gets low
# - Implemented fallback mechanisms for when parallel processing fails
#
# CCPEAKS_PLOT
# ============
# A tool for 3D template matching and peak analysis in cryo-electron tomography (cryoET).
#
# FEATURES:
#  1) Computes 3D cross-correlation (CC) maps between tomograms and one or more templates.
#  2) Uses an iterative non-maximum suppression (NMS) algorithm:
#       - Finds the highest voxel in the CC map,
#       - Masks out all voxels within --diameter (a spherical region) around that maximum,
#       - Repeats until the desired number of peaks are extracted per template.
#       - Later, the final aggregator selects the top --npeaks among all templates.
#  3) Multiple peak-finding methods:
#       - 'original': Reliable non-maximum suppression algorithm
#       - 'eman2': Uses EMAN2 for fast peak finding (recommended if available)
#       - 'vectorized': Optimized NumPy vectorized implementation for better performance
#       - 'fallback': Simple NMS implementation that works when EMAN2 is not available
#  4) Flexible CC map normalization methods:
#       - 'standard': Z-score normalization (subtract mean, divide by standard deviation)
#       - 'mad': Median absolute deviation normalization (robust to outliers)
#       - 'background': Background-based normalization using either:
#           * Edge-based approach: statistics from voxels near volume boundaries (default)
#           * Radius-based approach: statistics from voxels outside a central sphere
#  5) Flexible thresholding with --cc_thresh (default 0.0, disabled):
#       For example, --cc_thresh 3 means only consider voxels with CC >= (mean + 3×std)
#  6) Data comparison features:
#       - When two input files are provided, compares the CC distributions
#       - --match_sets_size ensures equal sample sizes from each dataset
#       - --subset limits processing to the first n images (memory efficient)
#       - Statistical comparison with p-values and effect sizes in the output plots
#  7) Comprehensive outputs:
#       - Saves per-template results in "correlation_files"
#       - Combines final peaks in "cc_peaks" directory
#       - Optionally saves full CC maps in EMAN2-compatible HDF format
#       - Generates distribution plots of average CC values and all peaks
#       - Optionally exports CSV summary with --save_csv
#       - Creates coordinate map visualizations with --save_coords_map and --coords_map_r
#
# PARAMETERS:
# ----------
# Required:
#   --input: Comma-separated tomogram file paths (max 2 allowed)
#   --template: Template file path (can be HDF stack with multiple templates)
#
# Important options:
#   --background_radius: For radius-based background normalization (mutually exclusive with --background_edge)
#   --background_edge: For edge-based background normalization (mutually exclusive with --background_radius)
#   --cc_norm_method: 'background', 'mad', or 'standard' (default='mad')
#   --coords_map_r: Radius for spheres in coordinate maps (default=3)
#   --diameter: Min allowed distance between peaks (auto-calculated if not provided)
#   --npeaks: Number of final peaks to keep (auto-calculated if not provided)
#   --peak_method: 'original', 'eman2', 'vectorized', or 'fallback' (default='original')
#
# USAGE (example):
#   python ccpeaks_plot.py --input tomo1.hdf,tomo2.hdf --template template.hdf \
#       --npeaks 2 --peak_method eman2 --cc_norm_method background \
#       --background_radius 21
#
# DEPENDENCIES:
#   - Required: NumPy, SciPy, matplotlib, tqdm, psutil, h5py, mrcfile
#   - Optional: EMAN2 (provides faster masking and peak detection)

import argparse
import os
import sys
import numpy as np
import mrcfile
import h5py
import matplotlib.pyplot as plt
from scipy.signal import correlate, fftconvolve
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import logging
import time
import traceback
import gc  # For explicit garbage collection
import math  # For calculating optimal batch sizes
from mpl_toolkits.mplot3d import Axes3D
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import psutil
import csv

# Enable multithreading in NumPy for faster array operations
# Setting this early ensures all NumPy operations benefit from multithreading
try:
    # Try to use all available cores for NumPy operations
    import mkl
    mkl.set_num_threads(psutil.cpu_count(logical=True))
except ImportError:
    # If MKL is not available, try to use OpenBLAS settings
    try:
        from numpy import __config__
        # Check if OpenBLAS is being used
        if hasattr(__config__, '_config') and 'openblas' in str(__config__._config).lower():
            os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
            os.environ["OPENBLAS_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
    except:
        # Fallback - just set environment variables
        os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
        os.environ["OPENBLAS_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
        os.environ["MKL_NUM_THREADS"] = str(psutil.cpu_count(logical=True))

try:
    from EMAN2 import EMData, EMNumPy
    EMAN2_AVAILABLE = True
except ImportError:
    EMAN2_AVAILABLE = False

_VERBOSE = False  # Global for controlling debug logs in iterative NMS

#########################
# Utility functions
#########################

def create_output_directory(base_dir):
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "correlation_files"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "coordinate_plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw_coordinates"), exist_ok=True)
    return output_dir

def setup_logging(output_dir, verbose_level):
    log_file = os.path.join(output_dir, "run.log")
    if verbose_level <= 0:
        log_level = logging.ERROR
    elif verbose_level == 1:
        log_level = logging.WARNING
    elif verbose_level == 2:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logging.info("Command: " + " ".join(sys.argv))
    return log_file

def parse_mrc_apix_and_mean(file_path):
    try:
        with mrcfile.open(file_path, permissive=True) as mrc:
            data = mrc.data
            if data is None:
                return (None, None)
            mean_val = data.mean()
            if hasattr(mrc, 'voxel_size') and mrc.voxel_size.x > 0:
                apix = float(mrc.voxel_size.x)
            else:
                nx = mrc.header.nx
                if nx > 0:
                    apix = float(mrc.header.cella.x / nx)
                else:
                    apix = None
            return (apix, mean_val)
    except:
        return (None, None)

def parse_hdf_apix_and_mean(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            apix = None
            mean_val = None
            if 'apix' in f.attrs:
                apix = float(f.attrs['apix'])
            elif 'MDF' in f.keys() and 'apix' in f['MDF'].attrs:
                apix = float(f['MDF'].attrs['apix'])
            if 'MDF/images' in f.keys():
                ds_keys = list(f['MDF/images'].keys())
                if ds_keys:
                    sums = 0.0
                    count = 0
                    for k in ds_keys:
                        if 'image' in f['MDF/images'][k]:
                            arr = f['MDF/images'][k]['image'][:]
                            sums += arr.sum()
                            count += arr.size
                    if count > 0:
                        mean_val = sums / count
            return (apix, mean_val)
    except:
        return (None, None)

def load_image(file_path, img_idx=None):
    """
    Load images from a file.
    
    Args:
        file_path: Path to the image file
        img_idx: If provided, loads only the specified image index.
                 If None, loads all images.
    
    Returns:
        A list of images if img_idx is None, or a single image if img_idx is specified.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.hdf', '.h5']:
            with h5py.File(file_path, 'r') as file:
                if img_idx is not None:
                    # Load only the requested image
                    dataset_path = f"MDF/images/{img_idx}/image"
                    return file[dataset_path][:]
                else:
                    # Load all images
                    dataset_paths = [f"MDF/images/{i}/image" for i in range(len(file["MDF/images"]))]
                    images = [file[ds][:] for ds in dataset_paths]
                    return images
        elif ext == '.mrc':
            with mrcfile.open(file_path, permissive=True) as mrc:
                data = mrc.data
                if img_idx is not None:
                    if data.ndim == 3:
                        # Only one image in the file
                        if img_idx == 0:
                            return data
                        else:
                            raise ValueError(f"Image index {img_idx} out of range for MRC file with one image")
                    elif data.ndim >= 4:
                        # Multiple images in the file
                        if img_idx < len(data):
                            return data[img_idx]
                        else:
                            raise ValueError(f"Image index {img_idx} out of range for MRC file with {len(data)} images")
                    else:
                        if img_idx == 0:
                            return data
                        else:
                            raise ValueError(f"Image index {img_idx} out of range for MRC file with one image")
                else:
                    # Return all images
                    if data.ndim == 3:
                        return [data]
                    elif data.ndim >= 4:
                        return [d for d in data]
                    else:
                        return [data]
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        logging.error(f"Error loading image from {file_path}: {str(e)}")
        if img_idx is not None:
            logging.error(f"Specified image index: {img_idx}")
        raise
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def check_apix(tomo_path, template_path):
    tomo_ext = os.path.splitext(tomo_path)[1].lower()
    template_ext = os.path.splitext(template_path)[1].lower()
    if tomo_ext == '.mrc':
        tomo_apix, _ = parse_mrc_apix_and_mean(tomo_path)
    elif tomo_ext in ['.hdf', '.h5']:
        tomo_apix, _ = parse_hdf_apix_and_mean(tomo_path)
    else:
        tomo_apix = None
    if template_ext == '.mrc':
        tpl_apix, _ = parse_mrc_apix_and_mean(template_path)
    elif template_ext in ['.hdf', '.h5']:
        tpl_apix, _ = parse_hdf_apix_and_mean(template_path)
    else:
        tpl_apix = None
    if tomo_apix and tpl_apix:
        ratio = tpl_apix / tomo_apix if tomo_apix != 0 else None
        if ratio and (ratio < 0.5 or ratio > 2.0):
            logging.warning(f"Template apix {tpl_apix:.2f} vs. tomogram apix {tomo_apix:.2f} differs by >2x.")

def adjust_template(template, target_shape):
    t_shape = template.shape
    new_template = np.zeros(target_shape, dtype=template.dtype)
    slices_target = []
    slices_template = []
    for i, (t_dim, tgt_dim) in enumerate(zip(t_shape, target_shape)):
        if t_dim <= tgt_dim:
            start_tgt = (tgt_dim - t_dim) // 2
            slices_target.append(slice(start_tgt, start_tgt + t_dim))
            slices_template.append(slice(0, t_dim))
        else:
            start_tpl = (t_dim - tgt_dim) // 2
            slices_target.append(slice(0, tgt_dim))
            slices_template.append(slice(start_tpl, start_tpl + tgt_dim))
    new_template[tuple(slices_target)] = template[tuple(slices_template)]
    return new_template


def compute_fft(volume):
    """
    Compute the FFT of a volume for efficient correlation.
    
    Args:
        volume: 3D NumPy array
        
    Returns:
        FFT of the volume
    """
    # Use single precision for FFT which is much faster and uses half the memory
    # compared to double precision, with minimal impact on accuracy
    volume_float32 = volume.astype(np.float32, copy=False)
    
    # Use numpy's optimized rfftn function which is more memory efficient
    # than a full FFT since it exploits the symmetry of real inputs
    return np.fft.rfftn(volume_float32)


def compute_cc_map_fft(image_fft, template_fft, shape):
    """
    Compute cross-correlation map using precomputed FFTs.
    
    This function computes the cross-correlation map directly using FFT multiplication,
    which is more efficient when correlating one image with multiple templates.
    
    Args:
        image_fft: FFT of the image
        template_fft: FFT of the template
        shape: Shape of the original volumes
        
    Returns:
        Cross-correlation map
    """
    # Check if input data is the right type for best performance
    if not np.issubdtype(image_fft.dtype, np.complex128) and not np.issubdtype(image_fft.dtype, np.complex64):
        image_fft = image_fft.astype(np.complex64)
    if not np.issubdtype(template_fft.dtype, np.complex128) and not np.issubdtype(template_fft.dtype, np.complex64):
        template_fft = template_fft.astype(np.complex64)
    
    # Compute cc_map using FFT multiplication (conjugate for cross-correlation)
    # Use in-place conjugate to avoid memory allocation
    conj_template_fft = np.conjugate(template_fft, out=template_fft.copy())
    
    # Perform the multiplication and inverse FFT
    cc_map = np.fft.irfftn(image_fft * conj_template_fft, shape)
    
    # Properly shift the result for correct correlation
    # Note: fftshift can be a bottleneck for large arrays
    # Using in-place version can help with memory usage
    cc_map = np.fft.fftshift(cc_map)
    
    return cc_map


def calculate_optimal_batch_size(template_count, input_count, volume_size_mb, threads):
    """
    Calculate optimal batch size based on memory constraints.
    
    Args:
        template_count: Number of templates
        input_count: Number of inputs
        volume_size_mb: Estimated size in MB for one volume
        threads: Number of parallel threads
        
    Returns:
        batch_size: Optimal batch size
        batch_mode: 'templates' or 'inputs' indicating what to batch
    """
    # Get available system memory
    total_system_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    
    # Get memory_factor from globals or use default
    memory_factor = globals().get('memory_factor', 0.6)  # Default to 0.6 if not set
    
    # Adjust memory usage based on memory_factor:
    # - memory_factor 0.3 = conservative (30% of available, 20% of total)
    # - memory_factor 0.6 = balanced (60% of available, 40% of total)
    # - memory_factor 0.8 = aggressive (80% of available, 60% of total)
    max_memory_usage = min(available_memory * memory_factor, total_system_memory * (memory_factor * 0.75))
    
    # Log memory information
    logging.info(f"Memory info: Total={total_system_memory:.2f} MB, Available={available_memory:.2f} MB")
    logging.info(f"Planning to use at most {max_memory_usage:.2f} MB for processing")
    
    # Account for Python process overhead (interpreter, loaded modules, etc.)
    base_process_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    logging.info(f"Current process memory: {base_process_memory:.2f} MB")
    
    # Each correlation requires:
    # 1 image + 1 template + 1 FFT image + 1 FFT template + 1 CC map + temp arrays ≈ 6x volume size
    # Add extra 20% margin for unexpected allocations
    memory_per_correlation = volume_size_mb * 7.2  # 6 * 1.2
    
    # Determine batch mode based on which dimension is larger
    if template_count > input_count:
        # Batch by templates (keep all inputs in memory)
        batch_mode = 'templates'
        total_items = template_count
        memory_fixed = input_count * volume_size_mb
    else:
        # Batch by inputs (keep all templates in memory)
        batch_mode = 'inputs'
        total_items = input_count
        memory_fixed = template_count * volume_size_mb
    
    # Calculate memory available for batch items
    memory_for_batch = max(0, max_memory_usage - base_process_memory - memory_fixed)
    
    # Log memory allocation plan
    logging.info(f"Batching by {batch_mode}: {total_items} items total")
    logging.info(f"Fixed memory for loaded data: {memory_fixed:.2f} MB")
    logging.info(f"Memory available for batch processing: {memory_for_batch:.2f} MB")
    logging.info(f"Estimated memory per correlation: {memory_per_correlation:.2f} MB")
    
    # Calculate how many items we can process at once
    # Use a more conservative formula - scale memory requirement by threads
    # This helps prevent memory spikes from parallel allocations
    memory_scaling_factor = 1.0 + (0.1 * min(threads, 8))  # Increase memory estimate by 10% per thread (up to 8)
    adjusted_memory_per_corr = memory_per_correlation * memory_scaling_factor
    
    items_per_batch = max(1, int(memory_for_batch / (adjusted_memory_per_corr * threads)))
    
    # Adjust for thread count (ensure each thread gets work)
    # but be more conservative with high thread counts
    batch_size = max(threads, min(items_per_batch * threads, total_items))
    
    # Cap batch size at a reasonable maximum
    batch_size = min(batch_size, 500)
    
    # Ensure batch size is at least 1
    batch_size = max(1, batch_size)
    
    # Log final decision
    logging.info(f"Selected batch size: {batch_size} with {threads} parallel threads")
    logging.info(f"This will process ~{batch_size * threads} correlations in parallel")
    
    return batch_size, batch_mode


def compute_background_stats(image, cc_norm_method="standard", background_radius=None, background_edge=None):
    """
    Calculate normalization statistics from an image based on the specified method.
    
    This function is separate from the CC calculation to allow calculating background
    statistics only once per image and reusing them for multiple templates.
    
    Parameters:
        image: 3D NumPy array of the tomogram.
        cc_norm_method: Normalization method ("standard", "mad", or "background").
        background_radius: Optional integer for radius-based background.
        background_edge: Optional integer for edge-based background.
        
    Returns:
        stats: Dictionary with normalization parameters (mean, std, median, mad)
    """
    stats = {}
    
    if cc_norm_method == "background":
        # Create the background mask based on the provided method
        nz, ny, nx = image.shape
        mask = None
        
        # Radius-based background (voxels outside a central sphere)
        if background_radius is not None:
            # Ensure background_radius is reasonable
            min_image_dim = min(image.shape)
            if background_radius < 8 or background_radius >= min_image_dim // 2:
                raise ValueError(f"--background_radius must be between 8 and {min_image_dim // 2 - 1} (half the smallest image dimension); got {background_radius}")
            
            # Create spherical mask
            center = np.array(image.shape) // 2
            Z, Y, X = np.indices(image.shape)
            mask = ((X - center[2])**2 + (Y - center[1])**2 + (Z - center[0])**2) >= (background_radius ** 2)
            
        # Edge-based background (voxels within a certain distance from any face)
        elif background_edge is not None:
            # Ensure background_edge is reasonable
            min_image_dim = min(image.shape)
            if background_edge < 2 or background_edge >= min_image_dim // 4:
                raise ValueError(f"--background_edge must be between 2 and {min_image_dim // 4 - 1} (1/4 of the smallest image dimension); got {background_edge}")
                
            # Create edge-based mask - voxels within background_edge of any face
            Z, Y, X = np.indices(image.shape)
            mask = (Z < background_edge) | (Z >= nz - background_edge) | \
                   (Y < background_edge) | (Y >= ny - background_edge) | \
                   (X < background_edge) | (X >= nx - background_edge)
                
        # Default: Edge-based background with thickness of 3 voxels
        else:
            # Default edge thickness is 3 voxels or 5% of smallest dimension (whichever is larger)
            default_edge = max(3, int(min(image.shape) * 0.05))
            default_edge = min(default_edge, min(image.shape) // 10)  # Cap at 10% of smallest dimension
            
            # Create edge-based mask with default thickness
            Z, Y, X = np.indices(image.shape)
            mask = (Z < default_edge) | (Z >= nz - default_edge) | \
                   (Y < default_edge) | (Y >= ny - default_edge) | \
                   (X < default_edge) | (X >= nx - default_edge)
        
        # Calculate background statistics from the selected mask
        if np.any(mask):
            bg_mean = np.mean(image[mask])
            bg_std = np.std(image[mask])
            logging.debug(f"Background stats from {np.sum(mask)} voxels: mean={bg_mean:.4f}, std={bg_std:.4f}")
            stats["bg_mean"] = bg_mean
            stats["bg_std"] = bg_std
        else:
            # Fallback if mask is empty (should never happen with our checks)
            bg_mean = np.mean(image)
            bg_std = np.std(image)
            logging.warning("Background mask is empty; using whole map statistics")
            stats["bg_mean"] = bg_mean
            stats["bg_std"] = bg_std
            
    elif cc_norm_method == "mad":
        # MAD normalization: calculate median and median absolute deviation
        median_val = np.median(image)
        mad_val = np.median(np.abs(image - median_val))
        mad_val *= 1.4826  # Scale to approximate standard deviation
        stats["median"] = median_val
        stats["mad"] = mad_val
        
    else:  # standard normalization
        # Z-score: calculate mean and standard deviation
        mean_val = np.mean(image)
        std_val = np.std(image)
        stats["mean"] = mean_val
        stats["std"] = std_val
    
    return stats


def compute_ncc_map_3d_single(image, template, cc_norm_method="standard", background_radius=None, background_edge=None, cc_clip_size=None, image_stats=None, image_fft=None, template_fft=None):
    """
    Compute the normalized cross-correlation (CC) map between 'image' and 'template' using FFT-based correlation.
    
    Normalization is applied according to the selected method:
      - "standard": Subtract mean and divide by standard deviation of the entire map.
      - "mad": Subtract the median and divide by the median absolute deviation (MAD) of the entire map.
      - "background": Calculate statistics from background voxels defined by either:
                      1) Voxels outside a central sphere (if background_radius is provided)
                      2) Voxels near the edges of the volume (if background_edge is provided)
                      3) Default edge-based approach if neither parameter is provided
    
    Parameters:
        image: 3D NumPy array of the tomogram.
        template: 3D NumPy array of the template.
        cc_norm_method: Normalization method ("standard", "mad", or "background").
        background_radius: Optional integer specifying the radius (in voxels) to use for computing
                           background statistics. Voxels outside this radius from the center are
                           considered background.
        background_edge: Optional integer specifying the thickness (in voxels) from the edges to 
                         use for computing background statistics. Voxels within this distance from
                         any face of the volume are considered background.
        cc_clip_size: Optional integer specifying the size of a central cubic region to focus
                      cross-correlation analysis on. If provided, only this central region will
                      be analyzed for peaks, while the full volume is still used for normalization.
        image_stats: Optional pre-computed normalization statistics for the image.
                     If provided, these will be used instead of recalculating.
        image_fft: Optional pre-computed FFT of the image for faster correlation.
        template_fft: Optional pre-computed FFT of the template for faster correlation.
    
    Returns:
        cc_map: The normalized cross-correlation map.
    """
    # Ensure template is the same size as image
    if image.shape != template.shape:
        template = adjust_template(template, image.shape)
    
    # Compute cross-correlation map using the most efficient method
    if image_fft is not None and template_fft is not None:
        # Use pre-computed FFTs for faster correlation
        cc_map = compute_cc_map_fft(image_fft, template_fft, image.shape)
    else:
        # Fall back to standard correlation
        cc_map = correlate(image, template, mode='same', method='fft')
    
    # Store the full CC map for normalization
    full_cc_map = cc_map.copy()
    
    # Apply cc_clip_size if provided to focus on a central region
    if cc_clip_size is not None:
        # Validate cc_clip_size
        min_image_dim = min(cc_map.shape)
        if cc_clip_size < 10 or cc_clip_size >= min_image_dim:
            raise ValueError(f"--cc_clip_size must be between 10 and {min_image_dim - 1} (smallest image dimension - 1); got {cc_clip_size}")
        
        # Calculate the slice indices for the central cubic region
        center = np.array(cc_map.shape) // 2
        half_size = cc_clip_size // 2
        
        # Create slices for the central region
        z_start, z_end = max(0, center[0] - half_size), min(cc_map.shape[0], center[0] + half_size)
        y_start, y_end = max(0, center[1] - half_size), min(cc_map.shape[1], center[1] + half_size)
        x_start, x_end = max(0, center[2] - half_size), min(cc_map.shape[2], center[2] + half_size)
        
        # Create a mask that is True outside the central region (to zero out these values later)
        mask_outside_clip = np.ones(cc_map.shape, dtype=bool)
        mask_outside_clip[z_start:z_end, y_start:y_end, x_start:x_end] = False
        
        logging.debug(f"Using cc_clip_size={cc_clip_size}. Central region: z={z_start}:{z_end}, y={y_start}:{y_end}, x={x_start}:{x_end}")

    # Use pre-computed statistics if provided, otherwise calculate them from the CC map
    if cc_norm_method == "background":
        if image_stats and "bg_mean" in image_stats and "bg_std" in image_stats:
            # Use pre-computed background statistics
            bg_mean = image_stats["bg_mean"]
            bg_std = image_stats["bg_std"]
            
            # Normalize using background statistics
            if bg_std < 1e-12:
                logging.warning("Background standard deviation is nearly zero; using value of 1.0")
                bg_std = 1.0
            cc_map = (cc_map - bg_mean) / bg_std
        else:
            # Calculate statistics from the full CC map
            # Create the background mask based on the provided method
            nz, ny, nx = full_cc_map.shape
            mask = None
        if background_radius is not None:
            # Ensure background_radius is reasonable
            min_image_dim = min(cc_map.shape)
            if background_radius < 8 or background_radius >= min_image_dim // 2:
                raise ValueError(f"--background_radius must be between 8 and {min_image_dim // 2 - 1} (half the smallest image dimension); got {background_radius}")
            
            # Create spherical mask
            center = np.array(cc_map.shape) // 2
            Z, Y, X = np.indices(cc_map.shape)
            mask = ((X - center[2])**2 + (Y - center[1])**2 + (Z - center[0])**2) >= (background_radius ** 2)
            
        # Edge-based background (voxels within a certain distance from any face)
        elif background_edge is not None:
            # Ensure background_edge is reasonable
            min_image_dim = min(cc_map.shape)
            if background_edge < 2 or background_edge >= min_image_dim // 4:
                raise ValueError(f"--background_edge must be between 2 and {min_image_dim // 4 - 1} (1/4 of the smallest image dimension); got {background_edge}")
                
            # Create edge-based mask - voxels within background_edge of any face
            Z, Y, X = np.indices(cc_map.shape)
            mask = (Z < background_edge) | (Z >= nz - background_edge) | \
                   (Y < background_edge) | (Y >= ny - background_edge) | \
                   (X < background_edge) | (X >= nx - background_edge)
                
        # Default: Edge-based background with thickness of 3 voxels
        else:
            # Default edge thickness is 3 voxels or 5% of smallest dimension (whichever is larger)
            default_edge = max(3, int(min(full_cc_map.shape) * 0.05))
            default_edge = min(default_edge, min(cc_map.shape) // 10)  # Cap at 10% of smallest dimension
            
            # Create edge-based mask with default thickness
            Z, Y, X = np.indices(cc_map.shape)
            mask = (Z < default_edge) | (Z >= nz - default_edge) | \
                   (Y < default_edge) | (Y >= ny - default_edge) | \
                   (X < default_edge) | (X >= nx - default_edge)
            
        # Calculate background statistics from the selected mask (using full_cc_map for stats)
        if np.any(mask):
            bg_mean = np.mean(full_cc_map[mask])
            bg_std = np.std(full_cc_map[mask])
            logging.debug(f"Background stats from {np.sum(mask)} voxels: mean={bg_mean:.4f}, std={bg_std:.4f}")
        else:
            # Fallback if mask is empty (should never happen with our checks)
            bg_mean = np.mean(full_cc_map)
            bg_std = np.std(full_cc_map)
            logging.warning("Background mask is empty; using whole map statistics")
            
        # Normalize using background statistics
        cc_map = (cc_map - bg_mean) / (bg_std if bg_std != 0 else 1)
        
    elif cc_norm_method == "mad":
        # MAD normalization: subtract median, divide by median absolute deviation
        if image_stats and "median" in image_stats and "mad" in image_stats:
            # Use pre-computed stats
            med = image_stats["median"]
            mad = image_stats["mad"]
        else:
            # Calculate from the CC map
            med = np.median(full_cc_map)
            mad = np.median(np.abs(full_cc_map - med))
            mad *= 1.4826  # Scale to approximate standard deviation
            
        cc_map = (cc_map - med) / (mad if mad != 0 else 1)
        
    else:  # standard normalization
        # Z-score normalization: subtract mean, divide by standard deviation
        if image_stats and "mean" in image_stats and "std" in image_stats:
            # Use pre-computed stats
            mean_val = image_stats["mean"]
            std_val = image_stats["std"]
        else:
            # Calculate from the CC map
            mean_val = np.mean(full_cc_map)
            std_val = np.std(full_cc_map)
            
        cc_map = (cc_map - mean_val) / (std_val if std_val != 0 else 1)
    
    # Apply clip mask if cc_clip_size was specified
    if cc_clip_size is not None:
        # Set all values outside the central region to -infinity to exclude them from peak finding
        cc_map[mask_outside_clip] = -np.inf
        logging.info(f"Applied cc_clip_size={cc_clip_size}: focusing analysis on central region while using full volume for normalization")
        
    return cc_map

#########################
# Local NMS methods
#########################

def numpy_to_emdata(a):
    return EMNumPy.numpy2em(a)

def non_maximum_suppression(cc_map, n_peaks, diameter):
    """
    A robust non-maximum suppression algorithm that returns the top n_peaks.
    The input cc_map is a NumPy array in [z,y,x] indexing.
    This function converts the candidate coordinates to (x,y,z) order (EMAN2 style)
    before returning them.
    
    Parameters:
        cc_map: 3D NumPy array of the CC values.
        n_peaks: Number of peaks to select.
        diameter: Minimum allowed distance between peaks (in voxels).
    
    Returns:
        List of tuples in (x, y, z, value) format.
    """
    flat = cc_map.ravel()
    sorted_indices = np.argsort(flat)[::-1]
    # np.unravel_index returns indices in (z, y, x) order.
    coords = np.column_stack(np.unravel_index(sorted_indices, cc_map.shape))
    accepted = []
    for idx, coord in enumerate(coords):
        candidate_val = flat[sorted_indices[idx]]
        if candidate_val == -np.inf:
            continue
        too_close = False
        # Convert candidate coordinate from (z,y,x) to (x,y,z)
        candidate_xyz = np.array([coord[2], coord[1], coord[0]])
        for (ax, ay, az, aval) in accepted:
            accepted_xyz = np.array([ax, ay, az])
            if np.linalg.norm(candidate_xyz - accepted_xyz) < diameter:
                too_close = True
                break
        if not too_close:
            accepted.append((int(candidate_xyz[0]), int(candidate_xyz[1]), int(candidate_xyz[2]), float(candidate_val)))
        if n_peaks > 0 and len(accepted) >= n_peaks:
            break
    return accepted


def vectorized_non_maximum_suppression(cc_map, n_peaks, diameter):
    """
    Highly optimized vectorized non-maximum suppression algorithm using NumPy.
    This function converts the coordinates from (z,y,x) order (as returned by np.unravel_index)
    to (x,y,z) order before returning them.
    
    Parameters:
        cc_map: 3D NumPy array of cross-correlation values.
        n_peaks: Maximum number of peaks to pick.
        diameter: Minimum allowed distance between peaks (in voxels).
    
    Returns:
        List of tuples in (x, y, z, value) format.
    """
    # Early exit if n_peaks is 0
    if n_peaks == 0:
        return []
    
    # Use faster flattening and avoid making unnecessary copies
    flat = cc_map.ravel()
    
    # Skip slow sorting operations if we're dealing with a large array
    if flat.size > 1_000_000:  # 100³ is 1 million elements
        # For large arrays, use argpartition which is much faster than argsort
        # Only get 20x the peaks we need (or all of them if size is smaller)
        candidate_count = min(20 * n_peaks, flat.size)
        top_indices = np.argpartition(flat, -candidate_count)[-candidate_count:]
        # Sort only these candidates (much faster than sorting the whole array)
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
    else:
        # For smaller arrays, finding all maximums is still efficient
        top_indices = np.argsort(flat)[::-1]
    
    # Convert to coordinates efficiently
    coords = np.asarray(np.unravel_index(top_indices, cc_map.shape)).T
    
    # Preallocate output for speed
    accepted = []
    # Process maxima in order
    diameter_squared = diameter * diameter
    
    # Special handling for first peak to avoid checks
    if coords.size > 0:
        # Convert candidate coordinate from (z,y,x) to (x,y,z)
        first_coord = coords[0]
        first_val = flat[top_indices[0]]
        # Skip if the value is -inf (was masked)
        if first_val != -np.inf:
            first_xyz = (int(first_coord[2]), int(first_coord[1]), int(first_coord[0]))
            accepted.append((first_xyz[0], first_xyz[1], first_xyz[2], float(first_val)))
    
    # Process remaining peaks, avoiding Python loops where possible
    accepted_coords = np.zeros((n_peaks, 3), dtype=np.int32)
    if accepted:
        accepted_coords[0] = [accepted[0][0], accepted[0][1], accepted[0][2]]
        acc_count = 1
    else:
        acc_count = 0
    
    # Process remaining coordinates
    for idx in range(1, len(coords)):
        if acc_count >= n_peaks:
            break
            
        coord = coords[idx]
        val = flat[top_indices[idx]]
        
        # Skip masked values
        if val == -np.inf:
            continue
            
        # Convert to (x,y,z) order
        xyz = np.array([coord[2], coord[1], coord[0]])
        
        # Fast vectorized distance check
        if acc_count > 0:
            # Calculate squared distances to all accepted peaks in one operation
            diffs = accepted_coords[:acc_count] - xyz
            sq_dists = np.sum(diffs * diffs, axis=1)
            
            # If any peak is too close, skip this coordinate
            if np.any(sq_dists < diameter_squared):
                continue
        
        # Add to accepted points
        accepted.append((int(xyz[0]), int(xyz[1]), int(xyz[2]), float(val)))
        accepted_coords[acc_count] = xyz
        acc_count += 1
    
    return accepted


def mask_out_region_xyz(arr, x0, y0, z0, diameter):
    nz, ny, nx = arr.shape
    dd = diameter**2
    radius = int(diameter)
    for z in range(max(0, z0 - radius), min(nz, z0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(ny, y0 + radius + 1)):
            for x in range(max(0, x0 - radius), min(nx, x0 + radius + 1)):
                dx = x - x0
                dy = y - y0
                dz = z - z0
                if (dx*dx + dy*dy + dz*dz) <= dd:
                    arr[z, y, x] = -np.inf
    return arr

def iterative_nms_eman2(cc_map_np, n_peaks, diameter, cc_thresh, verbose=False):
    from EMAN2 import EMNumPy, EMData
    if diameter is None:
        diameter = 35.0  # Default if not provided
    radius = diameter / 2.0
    radius_squared = radius * radius
    if verbose:
        print(f"[DEBUG] iterative_nms_eman2 start, n_peaks={n_peaks}, diameter={diameter:.1f}, radius={radius:.1f}")
    arr = cc_map_np.copy().astype(np.float32)
    shape_nz, shape_ny, shape_nx = arr.shape
    if cc_thresh > 0:
        val_thr = arr.mean() + cc_thresh * arr.std()
        arr = np.where(arr >= val_thr, arr, -np.inf)
    if (not n_peaks) or (n_peaks <= 0):
        n_peaks = arr.size
    accepted = []
    for i in range(n_peaks):
        em = EMNumPy.numpy2em(arr)
        max_val = em["maximum"]
        if max_val == -np.inf:
            break
        eman_x, eman_y, eman_z = em.calc_max_location()
        eman_x, eman_y, eman_z = int(eman_x), int(eman_y), int(eman_z)
        del em
        if not (0 <= eman_z < shape_nz and 0 <= eman_y < shape_ny and 0 <= eman_x < shape_nx):
            if verbose:
                print(f"[WARNING] EMAN2 gave out-of-bounds peak (x={eman_x},y={eman_y},z={eman_z}), ignoring.")
        else:
            accepted.append((eman_x, eman_y, eman_z, float(max_val)))
            if verbose:
                print(f"[DEBUG] EMAN2 peak {i+1}: (x={eman_x}, y={eman_y}, z={eman_z}), val={max_val:.3f}")
        z_min = max(0, eman_z - int(radius))
        z_max = min(shape_nz, eman_z + int(radius) + 1)
        y_min = max(0, eman_y - int(radius))
        y_max = min(shape_ny, eman_y + int(radius) + 1)
        x_min = max(0, eman_x - int(radius))
        x_max = min(shape_nx, eman_x + int(radius) + 1)
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if ((x - eman_x)**2 + (y - eman_y)**2 + (z - eman_z)**2) <= radius_squared:
                        arr[z, y, x] = -np.inf
        if verbose and i < 3:
            if arr[eman_z, eman_y, eman_x] != -np.inf:
                print(f"[ERROR] Masking failed at iteration {i+1}: value at peak {arr[eman_z, eman_y, eman_x]}")
            else:
                print(f"[DEBUG] Masking successful at iteration {i+1}")
    return accepted, arr

def fallback_iterative_nms(cc_map, n_peaks, diameter, cc_thresh, verbose=False):
    """
    Optimized fallback iterative non-maximum suppression algorithm.
    
    This implementation avoids creating unnecessary copies of data and uses
    more efficient NumPy operations for finding peaks and masking.
    
    Args:
        cc_map: 3D array of cross-correlation values
        n_peaks: Number of peaks to extract
        diameter: Minimum allowed distance between peaks
        cc_thresh: Optional threshold in sigma units
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (list of peak coordinates and values, masked array)
    """
    # Make copy only if needed (avoid if already a copy)
    if cc_map is None:
        return [], None
        
    # Work with 32-bit floats for better performance and memory usage
    if cc_map.dtype != np.float32:
        arr = cc_map.astype(np.float32, copy=True)
    else:
        arr = cc_map.copy()
    
    # Get dimensions
    nz, ny, nx = arr.shape
    
    # Apply threshold if requested - do it in-place to save memory
    if cc_thresh > 0:
        val_thr = arr.mean() + cc_thresh * arr.std()
        arr[arr < val_thr] = -np.inf
    
    # Handle incorrect n_peaks
    if n_peaks is None or n_peaks <= 0:
        n_peaks = min(1000, arr.size)  # Cap at a reasonable value to avoid excessive iterations
    
    # Pre-compute diameter squared for distance calculations
    diameter_squared = diameter * diameter
    
    # Pre-allocate output for speed
    accepted = []
    
    # Faster masking function with clipping and vectorization
    def fast_mask_region(arr, x0, y0, z0, diameter_squared):
        # Calculate mask bounds with clipping to array dimensions
        radius = int(np.ceil(diameter))
        z_min, z_max = max(0, z0 - radius), min(nz, z0 + radius + 1)
        y_min, y_max = max(0, y0 - radius), min(ny, y0 + radius + 1)
        x_min, x_max = max(0, x0 - radius), min(nx, x0 + radius + 1)
        
        # Create coordinate grids for the mask region (vectorized)
        z_grid, y_grid, x_grid = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Calculate squared distances from center
        distances = (x_grid - x0)**2 + (y_grid - y0)**2 + (z_grid - z0)**2
        
        # Create mask where distances <= diameter_squared
        mask = distances <= diameter_squared
        
        # Apply mask to the array
        arr[z_min:z_max, y_min:y_max, x_min:x_max][mask] = -np.inf
        
        return arr
    
    # Main peak finding loop with early termination
    for i in range(n_peaks):
        # Find maximum value and position
        max_val = arr.max()
        if max_val == -np.inf:
            break
            
        # Get position of maximum
        max_idx = arr.argmax()
        z0, y0, x0 = np.unravel_index(max_idx, arr.shape)
        
        # Store peak in (x,y,z) order with value
        accepted.append((int(x0), int(y0), int(z0), float(max_val)))
        
        if verbose and i < 10:  # Only print first 10 peaks to avoid console spam
            print(f"[DEBUG fallback] peak {i+1}: (x={x0}, y={y0}, z={z0}), val={max_val:.3f}")
        
        # Mask out region around the peak
        arr = fast_mask_region(arr, x0, y0, z0, diameter_squared)
    
    return accepted, arr

#########################
# Output saving functions
#########################

def plot_3d_peak_coordinates(output_dir, filename, template_str, image_idx, peak_coords):
    if np.asarray(peak_coords).size == 0:
        return
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peak_coords[:,0], peak_coords[:,1], peak_coords[:,2], marker='o')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Top CC Peak Locations")
    image_str = str(image_idx).zfill(3)
    out_png = os.path.join(output_dir, 'coordinate_plots', f'cc_{filename}_T{template_str}_I{image_str}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    logging.info(f"Saved 3D peak plot to {out_png}")

def place_sphere(volume, center, radius):
    """
    Places a sphere of given radius at specified center coordinates in a 3D volume.
    
    Parameters:
    ----------
    volume : 3D numpy array
        The volume to modify, typically a zero-filled array of shape (nz, ny, nx)
    center : tuple of int
        The (x, y, z) coordinates of the sphere's center in the volume
    radius : int
        The radius of the sphere in voxels
        
    Returns:
    -------
    None, modifies the volume in-place
    
    Notes:
    -----
    This function sets volume[z, y, x] = 1 for all points within the sphere.
    The function handles boundary checking to avoid index errors.
    """
    if radius <= 0:
        logging.warning(f"Invalid radius {radius} provided to place_sphere, must be > 0")
        return
    
    x0, y0, z0 = center
    nz, ny, nx = volume.shape
    r2 = radius**2
    
    logging.debug(f"Placing sphere at center ({x0}, {y0}, {z0}) with radius {radius} in volume of shape {volume.shape}")
    
    # Iterate within the bounding box of the sphere, with bounds checking
    for z in range(max(0, z0 - radius), min(nz, z0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(ny, y0 + radius + 1)):
            for x in range(max(0, x0 - radius), min(nx, x0 + radius + 1)):
                # Only set points that are within the sphere
                if (x - x0)**2 + (y - y0)**2 + (z - z0)**2 <= r2:
                    volume[z, y, x] = 1

def save_coords_map_as_hdf(coords_map_stack, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.attrs['appion_version'] = np.string_('ccpeaks_plot')
        f.attrs['MDF'] = np.string_('1')
        mdf = f.create_group('MDF')
        mdf.attrs['version'] = np.string_('1.0')
        images = mdf.create_group('images')
        for i in range(len(coords_map_stack)):
            img_group = images.create_group(str(i))
            img_group.create_dataset('image', data=coords_map_stack[i], dtype=coords_map_stack[i].dtype)
    logging.info(f"Saved EMAN2-compatible coords_map to {out_path}")

def save_cc_maps_stack_eman2(cc_maps_dict, out_path):
    """
    Use EMAN2 to save multiple CC maps in a single .hdf stack.
    Each map is written as one slice.
    Only maps that are not None will be saved.
    """
    if not EMAN2_AVAILABLE:
        logging.error("EMAN2 not available; cannot save cc maps in EMAN2 format.")
        return
    
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    sorted_keys = sorted(cc_maps_dict.keys())
    count = 0  # Count the number of maps actually saved
    for i, tmpl_idx in enumerate(sorted_keys):
        arr = cc_maps_dict[tmpl_idx]
        if arr is None:
            logging.warning(f"Template key {tmpl_idx} has a None CC map; skipping this entry.")
            continue
        # Perform conversion from numpy array to EMData
        try:
            em = EMNumPy.numpy2em(arr)
        except Exception as conv_ex:
            logging.error(f"Error converting array for key {tmpl_idx}: {conv_ex}")
            continue
        # Write the image using EMAN2 native method; the image index is i (or use count if you want contiguous numbering)
        em.write_image(out_path, count)
        count += 1
    logging.info(f"Saved {count} CC map(s) to {out_path} using EMAN2")


def save_cc_maps_stack(cc_maps_dict, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sorted_keys = sorted(cc_maps_dict.keys())
    with h5py.File(out_path, 'w') as f:
        f.attrs['appion_version'] = np.string_('ccpeaks_plot')
        f.attrs['MDF'] = np.string_('1')
        mdf = f.create_group('MDF')
        mdf.attrs['version'] = np.string_('1.0')
        images = mdf.create_group('images')
        for i, tmpl_idx in enumerate(sorted_keys):
            img_group = images.create_group(str(i))
            img_group.create_dataset('image', data=cc_maps_dict[tmpl_idx], dtype=cc_maps_dict[tmpl_idx].dtype)
    logging.info(f"Saved {len(sorted_keys)} CC map(s) to {out_path} using HDF5")

#########################
# Auto-diameter calculation
#########################

def compute_auto_diameter(template, tomo_shape):
    smoothed = gaussian_filter(template, sigma=1)
    thresh = smoothed.mean() + 2 * smoothed.std()
    binary = (smoothed >= thresh).astype(np.uint8)
    indices = np.argwhere(binary)
    if indices.size == 0:
        return 0
    min_indices = indices.min(axis=0)
    max_indices = indices.max(axis=0)
    spans = max_indices - min_indices + 1
    longest_span = spans.max()
    shortest_span = spans.min()
    average_span = (longest_span + shortest_span) / 2.0
    smallest_tomo_dim = min(tomo_shape)
    auto_diameter = min(average_span, smallest_tomo_dim)
    logging.info(f"Auto-diameter calculation: longest_span={longest_span}, shortest_span={shortest_span}, average={average_span:.1f}, final={auto_diameter:.1f}")
    return auto_diameter

#########################
# Worker functions and batch processing
#########################

def optimized_batch_processing(input_files, template_file, args, templates, input_images=None):
    """
    Process images and templates in optimized batches to maximize efficiency and control memory usage.
    
    This function implements the intelligent batching strategy:
    1. Determine whether to batch by inputs or templates based on which is larger
    2. Pre-compute FFTs to reuse across comparisons
    3. Compute background statistics once per image
    
    Args:
        input_files: List of input file paths
        template_file: Template file path
        args: Command line arguments
        templates: List of template volumes
        input_images: Optional pre-loaded input images
        
    Returns:
        List of processed results
    """
    # Calculate volume size for memory estimates
    if templates and len(templates) > 0:
        template_shape = templates[0].shape
        voxel_count = np.prod(template_shape)
        volume_size_mb = voxel_count * 4 / (1024 * 1024)  # 4 bytes per float32
    else:
        # Default size estimate if no templates
        volume_size_mb = 64  # MB
    
    n_templates = len(templates)
    
    # Load input images if not provided
    if input_images is None:
        input_images = []
        for f in input_files:
            images = load_image(f)
            if images is None or len(images) == 0:
                logging.error(f"Could not load file {f}")
                continue
            input_images.append(images)
    
    n_inputs = sum(len(imgs) for imgs in input_images)
    
    # Determine optimal batch size and mode
    if args.process_batches:
        batch_size, batch_mode = calculate_optimal_batch_size(
            n_templates, n_inputs, volume_size_mb, args.threads)
        
        logging.info(f"Optimized batching: mode={batch_mode}, batch_size={batch_size}")
        logging.info(f"Total tasks: {n_templates} templates × {n_inputs} inputs = {n_templates * n_inputs}")
        
        # Calculate memory estimate
        memory_estimate = n_inputs * n_templates * volume_size_mb * 5  # 5x for all data structures
        logging.info(f"Estimated total memory without batching: {memory_estimate:.1f} MB")
    else:
        batch_mode = 'none'
        batch_size = max(n_templates, n_inputs)
        logging.info("Processing without batching (all templates and inputs loaded simultaneously)")
    
    # Prepare common data structures
    results = []
    template_ffts = {}
    first_image_shape = input_images[0][0].shape
    
    # Pre-compute template FFTs that can be reused
    if batch_mode != 'templates':  # If not batching templates or no batching
        logging.info("Pre-computing template FFTs...")
        for tmpl_idx, template in enumerate(templates):
            # Ensure template is correct size
            if template.shape != first_image_shape:
                template = adjust_template(template, first_image_shape)
            # Compute and store FFT
            template_ffts[tmpl_idx] = compute_fft(template)
        logging.info(f"Finished pre-computing {len(template_ffts)} template FFTs")
    
    # Process based on batching strategy
    if batch_mode == 'templates':
        # Batch by templates (process all inputs with each template batch)
        results = process_by_template_batches(
            input_files, template_file, args, templates, input_images, 
            batch_size, template_ffts)
            
    elif batch_mode == 'inputs':
        # Batch by inputs (process all templates with each input batch)
        results = process_by_input_batches(
            input_files, template_file, args, templates, input_images, 
            batch_size, template_ffts)
            
    else:
        # No batching - process all at once
        results = process_all_at_once(
            input_files, template_file, args, templates, input_images, template_ffts)
    
    return results


def process_by_template_batches(input_files, template_file, args, templates, input_images, batch_size, template_ffts=None):
    """Process in batches of templates"""
    results = []
    total_batches = (len(templates) + batch_size - 1) // batch_size
    
    # Pre-compute image FFTs and stats (all inputs)
    logging.info("Pre-computing image FFTs and statistics...")
    image_ffts = {}
    image_stats = {}
    
    for file_idx, images in enumerate(input_images):
        filename = os.path.splitext(os.path.basename(input_files[file_idx]))[0]
        for img_idx, image in enumerate(images):
            # Compute and store image FFT
            image_ffts[(filename, img_idx)] = compute_fft(image)
            
            # Compute and store image statistics
            image_stats[(filename, img_idx)] = compute_background_stats(
                image, args.cc_norm_method, args.background_radius, args.background_edge)
    
    # Process templates in batches
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(templates))
        
        logging.info(f"Processing template batch {batch_idx+1}/{total_batches} (templates {batch_start}-{batch_end-1})")
        
        # Process this batch of templates against all inputs
        batch_tasks = []
        for tmpl_idx in range(batch_start, batch_end):
            template = templates[tmpl_idx]
            # Compute template FFT for this batch
            if tmpl_idx not in template_ffts:
                if template.shape != input_images[0][0].shape:
                    template = adjust_template(template, input_images[0][0].shape)
                template_ffts[tmpl_idx] = compute_fft(template)
                
            # Create tasks for all inputs with this template
            for file_idx, images in enumerate(input_images):
                filename = os.path.splitext(os.path.basename(input_files[file_idx]))[0]
                for img_idx in range(len(images)):
                    # Parameters for task
                    if args.npeaks is None:
                        diam_int = int(args.diameter)
                        n_possible = (images[img_idx].shape[0] // diam_int) * \
                                    (images[img_idx].shape[1] // diam_int) * \
                                    (images[img_idx].shape[2] // diam_int)
                        local_npeaks = n_possible
                    else:
                        local_npeaks = args.npeaks
                        
                    store_map = (tmpl_idx < args.save_cc_maps)
                    
                    # Create task with pre-computed data
                    task = (
                        filename, img_idx, tmpl_idx,
                        images[img_idx],  # Full image 
                        template,         # Full template
                        local_npeaks, args.ccc_thresh_sigma, args.diameter,
                        store_map, args.peak_method, args.cc_norm_method,
                        args.background_radius, args.background_edge, args.cc_clip_size,
                        image_ffts.get((filename, img_idx)),  # Pre-computed image FFT
                        template_ffts.get(tmpl_idx),          # Pre-computed template FFT
                        image_stats.get((filename, img_idx))  # Pre-computed image stats
                    )
                    batch_tasks.append(task)
        
        # Process this batch with parallel workers
        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_image_template_optimized, t) for t in batch_tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_idx+1}/{total_batches}"):
                result = fut.result()
                if result is not None:
                    results.append(result)
        
        # Clean up batch resources
        gc.collect()
        
    return results


def process_by_input_batches(input_files, template_file, args, templates, input_images, batch_size, template_ffts):
    """Process in batches of inputs"""
    results = []
    
    # Flatten input images for batching
    flat_inputs = []
    for file_idx, images in enumerate(input_images):
        filename = os.path.splitext(os.path.basename(input_files[file_idx]))[0]
        for img_idx, image in enumerate(images):
            flat_inputs.append((filename, file_idx, img_idx, image))
    
    total_batches = (len(flat_inputs) + batch_size - 1) // batch_size
    
    # Process inputs in batches
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(flat_inputs))
        
        logging.info(f"Processing input batch {batch_idx+1}/{total_batches} (inputs {batch_start}-{batch_end-1})")
        
        # Pre-compute FFTs and stats for this batch of inputs
        batch_image_ffts = {}
        batch_image_stats = {}
        
        for i in range(batch_start, batch_end):
            filename, _, img_idx, image = flat_inputs[i]
            # Compute and store image FFT
            batch_image_ffts[(filename, img_idx)] = compute_fft(image)
            
            # Compute and store image statistics
            batch_image_stats[(filename, img_idx)] = compute_background_stats(
                image, args.cc_norm_method, args.background_radius, args.background_edge)
        
        # Process all templates with this batch of inputs
        batch_tasks = []
        for tmpl_idx, template in enumerate(templates):
            # For each input in this batch
            for i in range(batch_start, batch_end):
                filename, _, img_idx, image = flat_inputs[i]
                
                # Parameters for task
                if args.npeaks is None:
                    diam_int = int(args.diameter)
                    n_possible = (image.shape[0] // diam_int) * \
                                (image.shape[1] // diam_int) * \
                                (image.shape[2] // diam_int)
                    local_npeaks = n_possible
                else:
                    local_npeaks = args.npeaks
                    
                store_map = (tmpl_idx < args.save_cc_maps)
                
                # Create task with pre-computed data
                task = (
                    filename, img_idx, tmpl_idx,
                    image,                              # Full image 
                    template,                           # Full template
                    local_npeaks, args.ccc_thresh_sigma, args.diameter,
                    store_map, args.peak_method, args.cc_norm_method,
                    args.background_radius, args.background_edge, args.cc_clip_size,
                    batch_image_ffts.get((filename, img_idx)),  # Pre-computed image FFT
                    template_ffts.get(tmpl_idx),                # Pre-computed template FFT
                    batch_image_stats.get((filename, img_idx))  # Pre-computed image stats
                )
                batch_tasks.append(task)
        
        # Process this batch with parallel workers
        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_image_template_optimized, t) for t in batch_tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_idx+1}/{total_batches}"):
                result = fut.result()
                if result is not None:
                    results.append(result)
        
        # Clean up batch resources
        gc.collect()
        
    return results


def process_all_at_once(input_files, template_file, args, templates, input_images, template_ffts):
    """Process all templates and inputs at once (no batching)"""
    results = []
    
    # Pre-compute all image FFTs and stats
    image_ffts = {}
    image_stats = {}
    
    for file_idx, images in enumerate(input_images):
        filename = os.path.splitext(os.path.basename(input_files[file_idx]))[0]
        for img_idx, image in enumerate(images):
            # Compute and store image FFT
            image_ffts[(filename, img_idx)] = compute_fft(image)
            
            # Compute and store image statistics
            image_stats[(filename, img_idx)] = compute_background_stats(
                image, args.cc_norm_method, args.background_radius, args.background_edge)
    
    # Create tasks for all template-image pairs
    all_tasks = []
    for tmpl_idx, template in enumerate(templates):
        for file_idx, images in enumerate(input_images):
            filename = os.path.splitext(os.path.basename(input_files[file_idx]))[0]
            for img_idx, image in enumerate(images):
                # Parameters for task
                if args.npeaks is None:
                    diam_int = int(args.diameter)
                    n_possible = (image.shape[0] // diam_int) * \
                                (image.shape[1] // diam_int) * \
                                (image.shape[2] // diam_int)
                    local_npeaks = n_possible
                else:
                    local_npeaks = args.npeaks
                    
                store_map = (tmpl_idx < args.save_cc_maps)
                
                # Create task with pre-computed data
                task = (
                    filename, img_idx, tmpl_idx,
                    image,  # Full image 
                    template,  # Full template
                    local_npeaks, args.ccc_thresh_sigma, args.diameter,
                    store_map, args.peak_method, args.cc_norm_method,
                    args.background_radius, args.background_edge, args.cc_clip_size,
                    image_ffts.get((filename, img_idx)),  # Pre-computed image FFT
                    template_ffts.get(tmpl_idx),  # Pre-computed template FFT
                    image_stats.get((filename, img_idx))  # Pre-computed image stats
                )
                all_tasks.append(task)
    
    # Process all tasks with parallel workers
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_image_template_optimized, t) for t in all_tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = fut.result()
            if result is not None:
                results.append(result)
    
    return results


def batch_process_tasks(all_tasks, batch_size, max_workers):
    """
    Process tasks in batches to control memory usage with robust failure handling.
    
    Args:
        all_tasks: List of all tasks to process
        batch_size: Number of tasks to process in each batch
        max_workers: Number of parallel workers to use
        
    Returns:
        List of results from all tasks
    """
    results = []
    
    # Calculate optimal batch size based on system memory and max_workers
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    system_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
    
    # More conservative memory limit - use at most 50% of available memory
    # This leaves more headroom for system and other processes
    max_memory_usage = min(available_memory * 0.5, system_memory * 0.3)
    
    # Estimate average task memory requirement from first few tasks
    # Assume each task needs memory proportional to the data size
    if all_tasks and len(all_tasks) > 0:
        sample_task = all_tasks[0]
        if len(sample_task) >= 5:  # Ensure task tuple has expected elements
            image = sample_task[3]
            template = sample_task[4]
            
            if hasattr(image, 'shape') and hasattr(template, 'shape'):
                # Calculate approximate memory per task
                image_bytes = image.size * image.itemsize
                template_bytes = template.size * template.itemsize
                
                # Each task needs ~7x the input data size for processing
                # (input, template, cc_map, FFTs, temporary arrays, etc.)
                bytes_per_task = (image_bytes + template_bytes) * 7  # Increased from 5x to 7x for safety
                mb_per_task = bytes_per_task / (1024 * 1024)
                
                # Adjust batch size based on actual data size
                # Be more conservative to avoid memory issues
                memory_based_batch_size = int(max_memory_usage / (mb_per_task * max(1, max_workers)))
                
                if memory_based_batch_size < batch_size:
                    logging.info(f"Adjusting batch size from {batch_size} to {memory_based_batch_size} based on memory constraints")
                    logging.info(f"Estimated memory per task: {mb_per_task:.2f} MB, Available memory: {max_memory_usage:.2f} MB")
                    batch_size = max(1, memory_based_batch_size)
    
    # Extra conservative - smaller batch size to avoid memory pressure
    # Start with smaller batches and gradually increase if things are working well
    batch_size = max(1, min(batch_size, max_workers * 2))
    
    # Cap at a reasonable maximum to prevent unlimited growth with large inputs
    batch_size = min(batch_size, 100)  # Reduced from 500 to be safer
    
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    logging.info(f"Processing {len(all_tasks)} tasks in {total_batches} batches of size {batch_size}")
    
    # Keep track of failures to handle potential full batch failures
    consecutive_failures = 0
    max_consecutive_failures = 3
    failed_task_indices = set()
    
    for i in range(0, len(all_tasks), batch_size):
        # Skip batches with indices past the end of all_tasks
        if i >= len(all_tasks):
            break
            
        # Get the current batch of tasks
        batch = all_tasks[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        # Log batch start
        logging.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} tasks")
        
        # Record memory before batch
        pre_batch_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # Use try/except to handle catastrophic failures in ProcessPoolExecutor
        try:
            batch_results = []
            
            # First try parallel processing
            if max_workers > 1 and consecutive_failures < max_consecutive_failures:
                try:
                    # Use fewer workers if we've had failures
                    effective_workers = max(1, max_workers - consecutive_failures) 
                    
                    # Create a process pool with timeout for worker initialization
                    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                        # Submit all tasks
                        futures = [executor.submit(process_image_template, t) for t in batch]
                        
                        # Process results as they complete
                        for fut_idx, fut in enumerate(tqdm(as_completed(futures), total=len(futures), 
                                                          desc=f"Batch {batch_num}/{total_batches}")):
                            try:
                                # Set a timeout to prevent hanging
                                result = fut.result(timeout=600)  # 10 minute timeout
                                if result is not None:
                                    batch_results.append(result)
                            except concurrent.futures.TimeoutError:
                                logging.error(f"Task {fut_idx} in batch {batch_num} timed out")
                            except Exception as e:
                                logging.error(f"Error processing task {fut_idx} in batch {batch_num}: {str(e)}")
                                logging.error(traceback.format_exc())
                    
                    # If we reach here, parallel processing worked
                    consecutive_failures = 0
                    
                except (concurrent.futures.process.BrokenProcessPool, OSError) as e:
                    # If ProcessPoolExecutor fails, fall back to sequential processing
                    logging.error(f"Process pool broken in batch {batch_num}, falling back to sequential processing: {str(e)}")
                    consecutive_failures += 1
                    batch_results = []  # Clear any partial results
                    
                    # Use sequential processing as fallback
                    for task_idx, task in enumerate(tqdm(batch, desc=f"Batch {batch_num} (sequential fallback)")):
                        try:
                            result = process_image_template(task)
                            if result is not None:
                                batch_results.append(result)
                        except Exception as e:
                            logging.error(f"Error processing task {task_idx} in sequential fallback: {str(e)}")
                            logging.error(traceback.format_exc())
            else:
                # Sequential processing from the start if workers=1 or too many failures
                logging.info(f"Using sequential processing for batch {batch_num} (workers={max_workers}, failures={consecutive_failures})")
                for task_idx, task in enumerate(tqdm(batch, desc=f"Batch {batch_num} (sequential)")):
                    try:
                        result = process_image_template(task)
                        if result is not None:
                            batch_results.append(result)
                    except Exception as e:
                        logging.error(f"Error processing task {task_idx} sequentially: {str(e)}")
                        logging.error(traceback.format_exc())
            
            # Append batch results to main results list
            results.extend(batch_results)
            
            # Log batch success
            logging.info(f"Successfully processed {len(batch_results)}/{len(batch)} tasks in batch {batch_num}")
            
            # If we had a good batch, cautiously increase batch size
            if len(batch_results) > 0.8 * len(batch) and consecutive_failures == 0:
                if batch_size < 100:  # Don't go too large
                    new_batch_size = min(int(batch_size * 1.2), 100)
                    if new_batch_size > batch_size:
                        logging.info(f"Increasing batch size from {batch_size} to {new_batch_size} due to good performance")
                        batch_size = new_batch_size
            
        except Exception as e:
            # Catch any other exceptions at the batch level
            logging.error(f"Critical error processing batch {batch_num}: {str(e)}")
            logging.error(traceback.format_exc())
            consecutive_failures += 1
            
            # If too many consecutive failures, reduce batch size dramatically
            if consecutive_failures >= max_consecutive_failures:
                new_batch_size = max(1, batch_size // 4)
                logging.warning(f"Too many consecutive failures. Reducing batch size from {batch_size} to {new_batch_size}")
                batch_size = new_batch_size
                consecutive_failures = 0  # Reset counter after taking action
        
        # Force garbage collection between batches
        gc.collect()
        
        # Check memory usage and log
        post_batch_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        memory_diff = post_batch_memory - pre_batch_memory
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        
        logging.info(f"Memory usage after batch {batch_num}: {post_batch_memory:.2f} MB (change: {memory_diff:+.2f} MB)")
        logging.info(f"System available memory: {available_memory:.2f} MB")
        
        # Dynamically adjust batch size if memory usage is too high
        if post_batch_memory > system_memory * 0.6 or available_memory < system_memory * 0.2:
            new_batch_size = max(1, int(batch_size * 0.5))  # Reduce by 50% if memory pressure is high
            logging.warning(f"High memory usage detected ({post_batch_memory:.2f} MB). Reducing batch size from {batch_size} to {new_batch_size}")
            batch_size = new_batch_size
        
    return results

def process_image_template_with_loading(task_info):
    """
    Process a task by first loading the required image/template data.
    
    Args:
        task_info: Tuple containing task information including file paths
        
    Returns:
        Result from processing the task
    """
    (filename, img_idx, tmpl_idx, img_path, img_index, 
     tmpl_path, tmpl_index, *other_args) = task_info
     
    # Load just the needed image and template
    image = load_image(img_path, img_idx=img_index)
    template = load_image(tmpl_path, img_idx=tmpl_index)
    
    # Process with loaded data
    result = process_image_template((filename, img_idx, tmpl_idx, 
                                    image, template, *other_args))
    
    # Help garbage collection
    del image
    del template
    
    return result


def process_image_template_optimized(args_tuple):
    """
    Optimized worker function that processes one (image, template) pair.
    
    Efficiently computes the CC map using precomputed FFTs and statistics.
    
    Args:
        args_tuple: Extended tuple with optional precomputed data
        
    Returns:
        Result tuple with peak information
    """
    try:
        # Unpack arguments, with support for pre-computed data
        if len(args_tuple) >= 16:  # Extended format with pre-computed data
            (filename, image_idx, tmpl_idx, image, template,
             n_peaks_local, cc_thresh, diameter, store_ccmap, method, cc_norm_method,
             background_radius, background_edge, cc_clip_size,
             image_fft, template_fft, image_stats) = args_tuple
        elif len(args_tuple) == 14:  # New format with cc_clip_size
            (filename, image_idx, tmpl_idx, image, template,
             n_peaks_local, cc_thresh, diameter, store_ccmap, method, cc_norm_method,
             background_radius, background_edge, cc_clip_size) = args_tuple
            image_fft = None
            template_fft = None
            image_stats = None
        else:  # Old format
            (filename, image_idx, tmpl_idx, image, template,
             n_peaks_local, cc_thresh, diameter, store_ccmap, method, cc_norm_method,
             background_radius, background_edge) = args_tuple
            cc_clip_size = None
            image_fft = None
            template_fft = None
            image_stats = None
            
        # Ensure diameter is not None
        if diameter is None:
            diameter = 35.0  # Default diameter if not specified
        
        # Use pre-computed statistics if available, otherwise compute them
        if image_stats is None:
            image_stats = compute_background_stats(
                image, 
                cc_norm_method=cc_norm_method,
                background_radius=background_radius,
                background_edge=background_edge
            )
        
        # Compute cross-correlation map with optimizations
        cc_map = compute_ncc_map_3d_single(
            image, 
            template, 
            cc_norm_method=cc_norm_method, 
            background_radius=background_radius,
            background_edge=background_edge,
            cc_clip_size=cc_clip_size,
            image_stats=image_stats,
            image_fft=image_fft,
            template_fft=template_fft
        )
        
        # Apply additional thresholding if requested
        if cc_thresh > 0:
            thresh = cc_map.mean() + cc_thresh * cc_map.std()
            cc_map = np.where(cc_map >= thresh, cc_map, -np.inf)
        
        # Find peaks using the appropriate method
        if method == "eman2" and EMAN2_AVAILABLE:
            # Use EMAN2 for peak finding
            local_peaks, cc_map_masked = iterative_nms_eman2(
                cc_map, n_peaks_local, diameter, cc_thresh, verbose=_VERBOSE
            )
            
            # Return with additional masked map for EMAN2 method
            if store_ccmap and tmpl_idx == 0:
                return (filename, image_idx, tmpl_idx, local_peaks, cc_map, cc_map_masked)
        
        elif method == "vectorized":
            # Use vectorized method
            local_peaks = vectorized_non_maximum_suppression(
                cc_map, n_peaks_local, diameter
            )
            
        elif method == "fallback" or (method == "eman2" and not EMAN2_AVAILABLE):
            # Use fallback method
            local_peaks, cc_map_masked = fallback_iterative_nms(
                cc_map, n_peaks_local, diameter, cc_thresh, verbose=_VERBOSE
            )
            
        else:
            # Use original method
            local_peaks = non_maximum_suppression(cc_map, n_peaks_local, diameter)
        
        # Return results
        if store_ccmap:
            return (filename, image_idx, tmpl_idx, local_peaks, cc_map)
        else:
            return (filename, image_idx, tmpl_idx, local_peaks)
            
    except Exception as e:
        logging.error(f"Error processing file {filename} image {image_idx} template {tmpl_idx}: {e}")
        logging.error(traceback.format_exc())
        return None

def process_image_template(args_tuple):
    """
    Worker function that processes one (image, template) pair.
    
    It computes the CC map using the requested normalization method and clip size,
    applies optional thresholding, and then extracts peaks using one of the available methods.
    
    Returns a tuple with the extracted peak information and optionally the raw CC map.
    
    This optimized version focuses on memory efficiency and improved error handling.
    """
    # Default values for error reporting
    filename = "unknown"
    image_idx = -1
    template_idx = -1
    
    try:
        # Handle multiple parameter formats with more flexibility
        if len(args_tuple) >= 16:  # Extended format with pre-computed data
            (filename, image_idx, template_idx, image, template,
             n_peaks_local, cc_thresh, diameter, store_ccmap, method,
             cc_norm_method, background_radius, background_edge, cc_clip_size,
             image_fft, template_fft, image_stats) = args_tuple[:17]
        elif len(args_tuple) >= 14:  # New format with cc_clip_size
            (filename, image_idx, template_idx, image, template,
             n_peaks_local, cc_thresh, diameter, store_ccmap, method,
             cc_norm_method, background_radius, background_edge, cc_clip_size) = args_tuple[:14]
            image_fft = None
            template_fft = None
            image_stats = None
        else:  # Old format
            (filename, image_idx, template_idx, image, template,
             n_peaks_local, cc_thresh, diameter, store_ccmap, method) = args_tuple[:10]
             
            # Set default values for optional parameters
            if len(args_tuple) > 10:
                cc_norm_method = args_tuple[10]
            else:
                cc_norm_method = "standard"
                
            if len(args_tuple) > 11:
                background_radius = args_tuple[11]
            else:
                background_radius = None
                
            if len(args_tuple) > 12:
                background_edge = args_tuple[12]
            else:
                background_edge = None
                
            cc_clip_size = None
            image_fft = None
            template_fft = None
            image_stats = None
            
        # Ensure diameter is not None
        if diameter is None:
            diameter = 35.0  # Default diameter if not specified
        
        # First make sure template and image are compatible sizes
        if image.shape != template.shape:
            template = adjust_template(template, image.shape)
        
        # Precompute image statistics for normalization if not provided
        if image_stats is None:
            image_stats = compute_background_stats(
                image, 
                cc_norm_method=cc_norm_method,
                background_radius=background_radius,
                background_edge=background_edge
            )
        
        # Compute FFTs if not provided
        if image_fft is None:
            image_fft = compute_fft(image)
        
        if template_fft is None:
            template_fft = compute_fft(template)
        
        # Compute CC map
        cc_map = compute_cc_map_fft(image_fft, template_fft, image.shape)
        
        # Explicitly release large temporary objects
        # Only delete FFTs if we created them (not if they were passed in)
        if image_fft is not None and (len(args_tuple) < 15 or args_tuple[14] is None):
            del image_fft
        if template_fft is not None and (len(args_tuple) < 16 or args_tuple[15] is None):
            del template_fft
        gc.collect()
        
        # Normalize CC map
        # Apply normalization directly here since normalize_cc_map isn't defined separately
        # Save memory by normalizing in place
        
        # Apply cc_clip_size if provided to focus on a central region
        mask_outside_clip = None
        if cc_clip_size is not None:
            # Validate cc_clip_size
            min_image_dim = min(cc_map.shape)
            if cc_clip_size < 10 or cc_clip_size >= min_image_dim:
                logging.warning(f"cc_clip_size {cc_clip_size} outside valid range (10 to {min_image_dim-1}); ignoring")
            else:
                # Calculate the slice indices for the central cubic region
                center = np.array(cc_map.shape) // 2
                half_size = cc_clip_size // 2
                
                # Create slices for the central region
                z_start, z_end = max(0, center[0] - half_size), min(cc_map.shape[0], center[0] + half_size)
                y_start, y_end = max(0, center[1] - half_size), min(cc_map.shape[1], center[1] + half_size)
                x_start, x_end = max(0, center[2] - half_size), min(cc_map.shape[2], center[2] + half_size)
                
                # Create a mask that is True outside the central region (to zero out these values later)
                mask_outside_clip = np.ones(cc_map.shape, dtype=bool)
                mask_outside_clip[z_start:z_end, y_start:y_end, x_start:x_end] = False
                
                logging.debug(f"Using cc_clip_size={cc_clip_size}. Central region: z={z_start}:{z_end}, y={y_start}:{y_end}, x={x_start}:{x_end}")
        
        # Normalize the CC map based on the specified method
        if cc_norm_method == "background":
            if image_stats and "bg_mean" in image_stats and "bg_std" in image_stats:
                # Use pre-computed background statistics
                bg_mean = image_stats["bg_mean"]
                bg_std = image_stats["bg_std"]
                
                # Normalize using background statistics
                if bg_std < 1e-12:
                    logging.warning("Background standard deviation is nearly zero; using value of 1.0")
                    bg_std = 1.0
                cc_map = (cc_map - bg_mean) / bg_std
            else:
                # Calculate on the fly - last resort if stats weren't precomputed
                bg_mean = np.mean(cc_map)
                bg_std = max(np.std(cc_map), 1e-12)
                cc_map = (cc_map - bg_mean) / bg_std
                
        elif cc_norm_method == "mad":
            # MAD normalization: subtract median, divide by median absolute deviation
            if image_stats and "median" in image_stats and "mad" in image_stats:
                # Use pre-computed stats
                med = image_stats["median"]
                mad = image_stats["mad"]
            else:
                # Calculate from the CC map
                med = np.median(cc_map)
                mad = np.median(np.abs(cc_map - med))
                mad *= 1.4826  # Scale to approximate standard deviation
                
            if mad < 1e-12:
                mad = 1.0
                logging.warning("MAD value is nearly zero; using value of 1.0")
                
            cc_map = (cc_map - med) / mad
            
        else:  # standard normalization
            # Z-score normalization: subtract mean, divide by standard deviation
            if image_stats and "mean" in image_stats and "std" in image_stats:
                # Use pre-computed stats
                mean_val = image_stats["mean"]
                std_val = image_stats["std"]
            else:
                # Calculate from the CC map
                mean_val = np.mean(cc_map)
                std_val = np.std(cc_map)
                
            if std_val < 1e-12:
                std_val = 1.0
                logging.warning("Standard deviation is nearly zero; using value of 1.0")
                
            cc_map = (cc_map - mean_val) / std_val
        
        # Apply clip mask if cc_clip_size was specified
        if mask_outside_clip is not None:
            # Set all values outside the central region to -infinity to exclude them from peak finding
            cc_map[mask_outside_clip] = -np.inf
            logging.info(f"Applied cc_clip_size={cc_clip_size}: focusing analysis on central region")
        
        # Apply thresholding if requested
        if cc_thresh > 0:
            thresh = cc_map.mean() + cc_thresh * cc_map.std()
            cc_map = np.where(cc_map >= thresh, cc_map, -np.inf)
            
        # Choose the peak-finding method based on the method parameter
        try:
            if method == "eman2" and EMAN2_AVAILABLE:
                local_peaks, cc_map_masked = iterative_nms_eman2(cc_map, n_peaks_local, diameter, cc_thresh, verbose=_VERBOSE)
                
                # Only return the masked map if explicitly requested to save memory
                if store_ccmap and template_idx == 0:
                    return (filename, image_idx, template_idx, local_peaks, cc_map, cc_map_masked)
            elif method == "vectorized":
                local_peaks = vectorized_non_maximum_suppression(cc_map, n_peaks_local, diameter)
                cc_map_masked = None
            elif method == "fallback" or (method == "eman2" and not EMAN2_AVAILABLE):
                if method == "eman2":
                    logging.warning(f"EMAN2 requested but not available, falling back to standard method")
                local_peaks, cc_map_masked = fallback_iterative_nms(cc_map, n_peaks_local, diameter, cc_thresh, verbose=_VERBOSE)
            else:  # default "original" method
                if cc_thresh > 0:
                    thresh_val = cc_map.mean() + cc_thresh * cc_map.std()
                    # Apply threshold in-place to save memory
                    cc_map[cc_map < thresh_val] = -np.inf
                    local_peaks = non_maximum_suppression(cc_map, n_peaks_local, diameter)
                else:
                    local_peaks = non_maximum_suppression(cc_map, n_peaks_local, diameter)
                cc_map_masked = None

            logging.debug(f"Found {len(local_peaks)} peaks for file {filename}, image {image_idx}, template {template_idx}")
            
            # Return only what's needed to minimize memory usage
            if store_ccmap:
                return (filename, image_idx, template_idx, local_peaks, cc_map)
            else:
                # Let the CC map be garbage collected immediately
                del cc_map
                gc.collect()
                return (filename, image_idx, template_idx, local_peaks)
                
        except MemoryError:
            logging.error(f"Memory error during peak finding for {filename}, image {image_idx}, template {template_idx}")
            # Try to recover some memory
            del cc_map
            gc.collect()
            # Return minimal result
            return (filename, image_idx, template_idx, [])
            
    except MemoryError:
        logging.error(f"Memory error processing {filename}, image {image_idx}, template {template_idx}")
        # Return empty peaks when we can't process an image
        return (filename, image_idx, template_idx, [])
    except Exception as e:
        logging.error(f"Error processing file {filename}, image {image_idx}, template {template_idx}: {str(e)}")
        logging.error(traceback.format_exc())
        return (filename, image_idx, template_idx, [])


#########################
# Final aggregator
#########################

def final_aggregator_greedy(candidate_list, n_peaks, min_distance):
    candidate_list.sort(key=lambda x: x[3], reverse=True)
    accepted = []
    for cand in candidate_list:
        if len(cand) >= 5:
            x, y, z, val, tmpl_idx = cand[:5]
        else:
            x, y, z, val = cand[:4]
            tmpl_idx = None
        too_close = False
        for accepted_peak in accepted:
            ax, ay, az = accepted_peak[:3]
            a_val = accepted_peak[3]
            dist = np.linalg.norm([x - ax, y - ay, z - az])
            if dist < min_distance:
                logging.debug(f"[DEBUG aggregator] Excluding peak (x={x}, y={y}, z={z}, val={val:.3f}) because distance {dist:.3f} < min_distance {min_distance}")
                too_close = True
                break
        if not too_close:
            accepted.append((x, y, z, val))
            logging.debug(f"[DEBUG aggregator] Accepting peak (x={x}, y={y}, z={z}, val={val:.3f})")
        if n_peaks > 0 and len(accepted) >= n_peaks:
            break
    return accepted

#########################
# Plot distribution
#########################

def plot_two_distributions(data1, data2, labels, output_path, title="CC Distribution"):
    # Handle empty data
    if len(data1) == 0:
        logging.warning(f"plot_two_distributions: data1 is empty, cannot create plot {output_path}")
        return
        
    data1 = np.asarray(data1)
    if data2 is not None and len(data2) > 0:
        data2 = np.asarray(data2)
    else:
        data2 = None
    if data2 is None or len(data2) == 0:
        fig, ax = plt.subplots(figsize=(8,8))
        parts = ax.violinplot([data1], positions=[1], showmeans=False, showmedians=False, showextrema=False)
        ax.scatter(np.full(data1.shape, 1), data1)
        q1, med, q3 = np.percentile(data1, [25,50,75])
        mean_val = np.mean(data1)
        ax.scatter(1, med, marker='s', color='red', zorder=3)
        ax.scatter(1, mean_val, marker='o', color='white', edgecolor='black', zorder=3)
        ax.vlines(1, data1.min(), data1.max(), color='black', lw=2)
        ax.vlines(1, q1, q3, color='black', lw=5)
        ax.set_xticks([1])
        ax.set_xticklabels(labels[:1])
        ax.set_ylabel("CC Values")
        ax.set_title(f"{title}\nN={len(data1)}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved single-distribution plot to {output_path}")
    else:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        def significance_from_pval(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            else: return "ns"
        def is_normal(dat):
            if len(dat) < 3:
                return False
            stat, p = shapiro(dat)
            return p > 0.05
        def calculate_effect_size(dat1, dat2):
            normal1 = is_normal(dat1)
            normal2 = is_normal(dat2)
            if normal1 and normal2:
                tstat, pval = ttest_ind(dat1, dat2, equal_var=False)
                mean1, mean2 = np.mean(dat1), np.mean(dat2)
                var1, var2 = np.var(dat1, ddof=1), np.var(dat2, ddof=1)
                n1, n2 = len(dat1), len(dat2)
                pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                effect = 0.0 if pooled_std < 1e-12 else (mean1 - mean2) / pooled_std
                method_label = "Cohen's d"
                if n1 >= 30 and n2 >= 30:
                    method_label = "z-score"
                return pval, effect, method_label
            else:
                stat, pval = mannwhitneyu(dat1, dat2, alternative='two-sided')
                n1, n2 = len(dat1), len(dat2)
                effect = 1 - (2*stat)/(n1*n2)
                return pval, effect, "Rank-biserial"
        pval, effect, method = calculate_effect_size(data1, data2)
        sig = significance_from_pval(pval)
        N1, N2 = len(data1), len(data2)
        fig, ax = plt.subplots(figsize=(8,8))
        parts = ax.violinplot([data1, data2], positions=[1,2], showmeans=False, showmedians=False, showextrema=False)
        colors = ['#1f77b4', '#ff7f0e']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.3)
        ax.scatter(np.full(data1.shape, 1), data1, edgecolor='k')
        q1_1, med1, q3_1 = np.percentile(data1, [25,50,75])
        mean1 = np.mean(data1)
        ax.scatter(1, med1, marker='s', color='red', zorder=3)
        ax.scatter(1, mean1, marker='o', color='white', edgecolor='black', zorder=3)
        ax.vlines(1, data1.min(), data1.max(), color='black', lw=2)
        ax.vlines(1, q1_1, q3_1, color='black', lw=5)
        ax.scatter(np.full(data2.shape, 2), data2, edgecolor='k')
        q1_2, med2, q3_2 = np.percentile(data2, [25,50,75])
        mean2 = np.mean(data2)
        ax.scatter(2, med2, marker='s', color='red', zorder=3)
        ax.scatter(2, mean2, marker='o', color='white', edgecolor='black', zorder=3)
        ax.vlines(2, data2.min(), data2.max(), color='black', lw=2)
        ax.vlines(2, q1_2, q3_2, color='black', lw=5)
        ax.set_xticks([1,2])
        ax.set_xticklabels(labels)
        ax.set_ylabel("CC Values")
        ax.set_title(f"{title}\n{labels[0]}: {N1} values, {labels[1]}: {N2} values\np={pval:.6f} ({sig}), {method}={effect:.4f}\nMean: {mean1:.3f} vs {mean2:.3f}, Median: {med1:.3f} vs {med2:.3f}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved two-distribution plot to {output_path}")

#########################
# Main function
#########################
def main():
    from collections import defaultdict
    parser = argparse.ArgumentParser(
        description=(
            "CCPEAKS_PLOT: 3D Template Matching and Peak Analysis for Cryo-ET\n\n"
            "This tool computes 3D cross-correlation (CC) maps between tomograms and templates,\n"
            "extracts peaks, and performs statistical analysis. It offers multiple normalization\n"
            "and peak-finding methods, with options for data comparison and visualization.\n\n"
            "Parameters (grouped by function):\n\n"
            "INPUT/OUTPUT:\n"
            "  --input: Required; comma-separated tomogram file paths; max 2 allowed.\n"
            "  --template: Required; template file path (HDF stack allowed).\n"
            "  --output_dir: Base output directory. Default='cc_analysis'.\n"
            "  --save_csv: If set, exports a CSV summary of peaks. Default=False.\n"
            "  --data_labels: Comma-separated labels for datasets. Default='file1,file2'.\n\n"
            "PROCESSING CONTROL:\n"
            "  --match_sets_size: If set, automatically match sets to the size of the smallest set when two input files are provided.\n"
            "  --subset: Process only the first n images from each input file. Use this to set a specific subset size.\n"
            "  --threads: Number of parallel processes. Default=1.\n"
            "  --verbose: Verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG). Default=2.\n\n"
            "NORMALIZATION & PEAK DETECTION:\n"
            "  --cc_norm_method: Normalization method ('background', 'mad', 'standard'). Default='mad'.\n"
            "  --background_radius: Radius for radius-based background statistics. Voxels outside this\n"
            "       radius from the center are considered background. Must be >8 and smaller than\n"
            "       half the smallest image dimension. Mutually exclusive with --background_edge.\n"
            "  --background_edge: Thickness for edge-based background statistics. Voxels within this\n"
            "       distance from any face are considered background. Must be >2 and smaller than\n"
            "       1/4 of the smallest image dimension. Mutually exclusive with --background_radius.\n"
            "       If neither option is provided but using 'background' method, edge-based approach\n"
            "       with an adaptive thickness is used by default.\n"
            "  --cc_thresh: Peak value threshold in sigma units (0.0=disabled). Default=0.0.\n"
            "  --peak_method: Algorithm to use ('original', 'eman2', 'vectorized', 'fallback'). Default='original'.\n"
            "  --diameter: Minimum allowed distance (voxels) between final peaks. Auto-derived if not provided.\n"
            "  --npeaks: Maximum number of final peaks per image. Auto-derived if not provided.\n\n"
            "VISUALIZATION:\n"
            "  --save_ccmaps: Number of CC maps to save per input file. Default=1.\n"
            "  --save_coords_map: Number of coordinate maps to save per input file. Default=1.\n"
            "  --coords_map_r: Radius (in voxels) for spheres in coordinate maps. Default=3.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "EXAMPLES:\n\n"
            "Basic usage with EMAN2 peak detection and radius-based background normalization:\n"
            "  python ccpeaks_plot.py --input tomo1.hdf,tomo2.hdf --template template.hdf \\\n"
            "      --npeaks 2 --peak_method eman2 --cc_norm_method background \\\n"
            "      --background_radius 21\n\n"
            "Using edge-based background normalization:\n"
            "  python ccpeaks_plot.py --input tomo1.hdf,tomo2.hdf --template template.hdf \\\n"
            "      --npeaks 2 --peak_method eman2 --cc_norm_method background \\\n"
            "      --background_edge 5\n\n"
            "Comparing two datasets with custom labels and visualization options:\n"
            "  python ccpeaks_plot.py --input clean.hdf,noisy.hdf --template ref.hdf \\\n"
            "      --data_labels=\"Clean,Noisy\" --peak_method vectorized --cc_norm_method mad \\\n"
            "      --save_ccmaps 3 --save_coords_map 5 --coords_map_r 4 --save_csv\n\n"
            "Processing a large dataset with memory optimization:\n"
            "  python ccpeaks_plot.py --input large_tomo.hdf --template ref.hdf \\\n"
            "      --subset 10 --threads 8 --verbose 3\n"
        )
    )

    # Create a mutually exclusive group for background definition methods
    background_group = parser.add_mutually_exclusive_group()
    background_group.add_argument('--background_radius', type=int, default=None,
                        help="Radius (in voxels) for calculating background statistics when using cc_norm_method 'background'.\n"
                             "Creates a spherical mask where voxels outside the radius are considered background.\n"
                             "Must be between 8 and half the smallest image dimension. Mutually exclusive with --background_edge.")
    background_group.add_argument('--background_edge', type=int, default=None,
                        help="Thickness (in voxels) from the edge to use for background statistics when using cc_norm_method 'background'.\n"
                             "Creates a shell mask where voxels within this distance from any face are considered background.\n"
                             "Must be between 2 and 1/4 of the smallest image dimension. Mutually exclusive with --background_radius.")
    
    # Other parameters (alphabetically ordered)
    parser.add_argument('--cc_clip_size', type=int, default=None,
                        help="Default=None (use full volume). Size of central cubic region to focus cross-correlation analysis on. \n"
                             "If provided, only analyzes peaks within this central region while still using the full volume\n"
                             "for background normalization. Useful when you want to normalize using background beyond the\n"
                             "particle but focus analysis on a specific interior region.")
    parser.add_argument('--cc_norm_method', type=str, default="mad", choices=["background", "mad", "standard"],
                        help="Default='mad'. CC normalization method. Use 'background' to use background voxels for normalization.")
    parser.add_argument('--cc_thresh', dest='ccc_thresh_sigma', type=float, default=0.0,
                        help="Default=0.0 (disabled). Threshold for picking peaks in sigma units above the mean.")
    parser.add_argument('--coords_map_r', type=int, default=3, 
                        help="Default=3. Radius (in voxels) for spheres in coordinate maps.")
    parser.add_argument('--data_labels', type=str, default="file1,file2",
                        help="Default='file1,file2'. Comma-separated labels for datasets.")
    parser.add_argument('--diameter', type=float, default=None,
                        help="Default=None (auto). Minimum allowed distance (voxels) between final peaks.")
    parser.add_argument('--process_batches', action='store_true', default=False,
                        help="Default=False. Process in memory-efficient batches. Reduces peak memory usage but may be slower.")
    parser.add_argument('--performance_mode', choices=['memory', 'balanced', 'speed'], default='balanced',
                        help="Default='balanced'. Performance optimization mode: 'memory' (lowest memory usage), 'balanced' (balanced speed and memory), or 'speed' (fastest but higher memory usage).")
    parser.add_argument('--input', required=True,
                        help="Required. Comma-separated tomogram file paths; maximum 2 allowed.")
    parser.add_argument('--match_sets_size', action='store_true', default=False,
                        help="Default=False. If set, automatically match sets to the size of the smallest set when two input files are provided.")
    parser.add_argument('--npeaks', type=int, default=None,
                        help="Default=None (auto). Number of final peaks to keep per image.")
    parser.add_argument('--output_dir', default="cc_analysis",
                        help="Default='cc_analysis'. Base output directory.")
    parser.add_argument('--peak_method', type=str, default="vectorized",
                        help="Default='vectorized'. Peak-finding method; options: 'vectorized' (fast), 'original', 'eman2' (if available), 'fallback'.")
    parser.add_argument('--save_cc_maps', type=int, default=1,
                        help="Default=1. Number of CC maps to save per input file.")
    parser.add_argument('--save_coords_maps', type=int, default=1,
                        help="Default=1. Number of coordinate maps to save per input file.")
    parser.add_argument('--save_csv', action='store_true', default=False,
                        help="Default=False. Export a CSV summary of extracted peaks.")
    parser.add_argument('--subset', type=int, default=None,
                        help="Default=None. Process only the first n images from each input file. Use this to set a specific subset size.")
    parser.add_argument('--template', required=True,
                        help="Required. Template file path (HDF allowed).")
    parser.add_argument('--threads', type=int, default=1,
                        help="Default=1. Number of parallel processes.")
    parser.add_argument('--verbose', type=int, default=2,
                        help="Default=2. Verbosity level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG.")

    args = parser.parse_args()

    # Early validations:
    if args.save_cc_maps < 1:
        logging.error("--save_cc_maps must be >= 1.")
        sys.exit(1)
    if args.save_coords_maps < 1:
        logging.error("--save_coords_maps must be >= 1.")
        sys.exit(1)

    input_files = args.input.split(',')
    if len(input_files) > 2:
        logging.error("Maximum of 2 input files allowed; please supply at most two.")
        sys.exit(1)

    global _VERBOSE
    _VERBOSE = (args.verbose >= 3)

    output_dir = create_output_directory(args.output_dir)
    setup_logging(output_dir, args.verbose)

    overall_start = time.time()
    logging.info("Starting processing...")
    logging.info(f"EMAN2 is {'available' if EMAN2_AVAILABLE else 'not available'}")
    
    # Apply performance mode settings
    if args.performance_mode == 'speed':
        # Speed optimization:
        # 1. Always use vectorized method if not explicitly overridden
        if args.peak_method == 'original':
            args.peak_method = 'vectorized'
            logging.info("Performance mode 'speed': Using vectorized peak finding method")
        # 2. Use more aggressive batching for better parallel processing
        memory_factor = 0.8  # Use more available memory
        # 3. Force process_batches to False unless memory is likely to be a constraint
        estimated_memory = psutil.virtual_memory().available / (1024*1024*1024)  # GB
        if estimated_memory > 16 and not args.process_batches:
            args.process_batches = False
            logging.info(f"Performance mode 'speed': Processing without batching (available memory: {estimated_memory:.1f} GB)")
        # 4. Use more threads by default if not specified
        if args.threads == 1:
            recommended_threads = min(psutil.cpu_count(logical=False) or 4, 8)
            args.threads = recommended_threads
            logging.info(f"Performance mode 'speed': Using {args.threads} threads")
            
    elif args.performance_mode == 'memory':
        # Memory optimization:
        # 1. Always use batching
        if not args.process_batches:
            args.process_batches = True
            logging.info("Performance mode 'memory': Enabling batch processing")
        # 2. Use smaller, more conservative batch sizes
        memory_factor = 0.3  # Use less memory per batch
        # 3. Limit threads to reduce memory pressure
        if args.threads > 4:
            args.threads = 4
            logging.info("Performance mode 'memory': Limiting to 4 threads to reduce memory usage")
    else:  # balanced mode
        # Balance memory and speed:
        memory_factor = 0.6  # Moderate memory usage
        # Use batching if memory is constrained
        total_memory = psutil.virtual_memory().total / (1024*1024*1024)  # GB
        if total_memory < 16 and not args.process_batches:
            args.process_batches = True
            logging.info(f"Performance mode 'balanced': Enabling batch processing (system memory: {total_memory:.1f} GB)")
        # Suggest optimal thread count
        if args.threads == 1:
            recommended_threads = min(max(2, psutil.cpu_count(logical=False) or 2), 6)
            args.threads = recommended_threads
            logging.info(f"Performance mode 'balanced': Using {args.threads} threads")
    
    logging.info(f"Performance configuration: mode={args.performance_mode}, threads={args.threads}, batch_process={args.process_batches}, peak_method={args.peak_method}")

    # Load templates.
    templates_list = load_image(args.template)
    if (templates_list is None) or (len(templates_list) == 0):
        logging.error("Could not load template file. Exiting.")
        sys.exit(1)
    templates = list(templates_list)
    n_templates = len(templates)
    logging.info(f"Loaded {n_templates} template(s) from {args.template}")

    if args.save_cc_maps > n_templates:
        logging.info(f"--save_ccmaps ({args.save_cc_maps}) is greater than the number of templates ({n_templates}); reducing to {n_templates}.")
        args.save_cc_maps = n_templates

    # Initialize variables
    image_counts = []
    input_images = []
    tasks = []
    shape_by_file_img = {}
    
    # Load all images
    for i, f in enumerate(input_files):
        check_apix(f, args.template)
        
        # Load images
        images = load_image(f)
        if images is None or len(images) == 0:
            logging.error(f"Could not load images from {f}. Exiting.")
            sys.exit(1)
            
        num_images = len(images)
        image_counts.append(num_images)
        input_images.append(images)
    
    # After loading input images, determine the smallest dimension across all input images.
    all_dims = []
    for file_images in input_images:
        for im in file_images:
            all_dims.append(min(im.shape))
    
    if not all_dims:
        logging.error("No images loaded!")
        sys.exit(1)
    
    global_min_dim = min(all_dims)
    
    # Get first image shape for auto-diameter calculation
    first_tomo_shape = input_images[0][0].shape
    
    # Auto-calculate diameter if not provided
    if args.diameter is None:
        auto_diameter = compute_auto_diameter(templates[0], first_tomo_shape)
        args.diameter = auto_diameter
        logging.info(f"No diameter provided; using auto-derived diameter: {auto_diameter}")

    # Build the task list using the loaded images
    tasks = []
    for i, images in enumerate(input_images):
        filename = os.path.splitext(os.path.basename(input_files[i]))[0]
        
        for img_idx, image in enumerate(images):
            shape_by_file_img[(filename, img_idx)] = image.shape
            
            # Calculate local_npeaks
            if args.npeaks is None:
                if args.diameter is None:
                    logging.error("Diameter is None and required for auto-calculating npeaks")
                    sys.exit(1)
                diam_int = int(args.diameter)
                n_possible = (image.shape[0] // diam_int) * (image.shape[1] // diam_int) * (image.shape[2] // diam_int)
                local_npeaks = n_possible
            else:
                local_npeaks = args.npeaks
            
            # Create tasks for each template
            for tmpl_idx, template_data in enumerate(templates):
                store_map = (tmpl_idx < args.save_cc_maps)
                tasks.append((
                    filename,
                    img_idx,
                    tmpl_idx,
                    image,
                    template_data,
                    local_npeaks,
                    args.ccc_thresh_sigma,
                    args.diameter,
                    store_map,
                    args.peak_method,
                    args.cc_norm_method,
                    args.background_radius,
                    args.background_edge,
                    args.cc_clip_size
                ))

    # right after loading all input_images and computing global_min_dim:
    if args.cc_norm_method == "background":
        # Calculate some bounds based on the smallest image dimension
        min_image_dim = global_min_dim
        
        # Handle background radius settings
        if args.background_radius is not None:
            # Fixed minimum and maximum for the radius approach
            min_bg_radius = 8  # Minimum sensible radius for background stats
            max_bg_radius = min_image_dim // 2  # Half the smallest dimension
            
            # Enforce bounds on background_radius
            if not (min_bg_radius <= args.background_radius < max_bg_radius):
                logging.error(
                    f"--background_radius must be between {min_bg_radius} and {max_bg_radius-1}; "
                    f"got {args.background_radius}"
                )
                sys.exit(1)
            
            logging.info(f"Using radius-based background with radius={args.background_radius}")
            
        # Handle background edge settings
        elif args.background_edge is not None:
            # Fixed minimum and maximum for the edge approach
            min_edge = 2  # Minimum sensible edge thickness
            max_edge = min_image_dim // 4  # Quarter of the smallest dimension
            
            # Enforce bounds on background_edge
            if not (min_edge <= args.background_edge < max_edge):
                logging.error(
                    f"--background_edge must be between {min_edge} and {max_edge-1}; "
                    f"got {args.background_edge}"
                )
                sys.exit(1)
                
            logging.info(f"Using edge-based background with thickness={args.background_edge}")
            
        # Default: Use edge-based approach with adaptive thickness
        else:
            # Default edge thickness is 3 voxels or 5% of smallest dimension (whichever is larger)
            default_edge = max(3, int(min_image_dim * 0.05))
            default_edge = min(default_edge, min_image_dim // 10)  # Cap at 10% of smallest dimension
            
            args.background_edge = default_edge
            logging.info(f"No background parameters provided; defaulting to edge-based with thickness={default_edge}")
            
        # cc_clip_size is now independent of background normalization
        # It can be used to focus analysis on the central region regardless of normalization method

    first_tomo_shape = input_images[0][0].shape

    if len(input_files) == 2 and args.match_sets_size:
        # Match sets to the size of the smallest set
        new_count = min(len(input_images[0]), len(input_images[1]))
        input_images[0] = input_images[0][:new_count]
        input_images[1] = input_images[1][:new_count]
        logging.info(f"Using --match_sets_size: processing {new_count} images from each input file (size of smallest set).")

    if args.subset is not None:
        new_count = min(*(len(x) for x in input_images), args.subset)
        for i in range(len(input_images)):
            input_images[i] = input_images[i][:new_count]
        logging.info(f"Using --subset: processing first {new_count} images from each input file.")

    # cc_clip_size is now independent of background normalization

    logging.info(f"Starting memory usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB")

    # Calculate memory requirements
    if templates and len(templates) > 0:
        template_shape = templates[0].shape
        voxel_count = np.prod(template_shape)
        volume_size_mb = voxel_count * 4 / (1024 * 1024)  # 4 bytes per float32
    else:
        # Default size estimate if no templates
        volume_size_mb = 64  # MB

    n_templates = len(templates)
    n_inputs = sum(len(imgs) for imgs in input_images)
    
    # Each correlation requires approximately 5x the volume size in memory
    memory_per_correlation = volume_size_mb * 5
    estimated_total_memory = n_templates * n_inputs * memory_per_correlation
    
    # Get available system memory (leave 20% for OS and other processes)
    total_system_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
    available_memory = total_system_memory * 0.8
    
    # Determine if memory-constrained batching is needed
    memory_constrained_batching = args.process_batches or (estimated_total_memory > available_memory * 0.7)
    
    if memory_constrained_batching:
        if not args.process_batches:
            logging.info(f"Automatically enabling memory-efficient batching: estimated memory usage ({estimated_total_memory:.1f} MB) exceeds 70% of available memory ({available_memory * 0.7:.1f} MB)")
        
        # Calculate optimal batch size based on memory constraints
        batch_size, batch_mode = calculate_optimal_batch_size(n_templates, n_inputs, volume_size_mb, args.threads)
        logging.info(f"Using memory-efficient batching: mode={batch_mode}, batch_size={batch_size}")
    else:
        # When not using memory-efficient batching, process all data at once but still use task batches for parallelism
        logging.info("Processing without memory constraints (all templates and inputs loaded simultaneously)")
        batch_size = min(len(tasks), args.threads * 8)  # Larger batches for better parallelism
        
    # We always use batch_process_tasks for parallel execution, but the meaning of "batch" is different:
    # - In memory-constrained mode: each batch loads/unloads data to control memory usage
    # - In non-constrained mode: batches are just for parallel task management
    if memory_constrained_batching:
        logging.info(f"Processing {len(tasks)} total tasks in memory-efficient batches of {batch_size}")
    else:
        logging.info(f"Processing {len(tasks)} total tasks with {batch_size} tasks per parallel batch")
        
    partial_results = batch_process_tasks(tasks, batch_size, args.threads)

    partial_peaks = defaultdict(list)
    cc_maps = {}
    for res in partial_results:
        if len(res) == 6:  # When using EMAN2 method, expect a 6-tuple (raw and masked CC maps)
            filename, image_idx, tmpl_idx, local_peaks, cc_map_raw, cc_map_masked = res
            cc_maps[(filename, image_idx, tmpl_idx)] = (cc_map_raw, cc_map_masked)
            logging.info(f"Saved both raw and masked CC maps for {filename} image {image_idx} template {tmpl_idx}")
        elif len(res) == 5:
            filename, image_idx, tmpl_idx, local_peaks, cc_map_raw = res
            cc_maps[(filename, image_idx, tmpl_idx)] = cc_map_raw
        else:
            filename, image_idx, tmpl_idx, local_peaks = res
        template_annotated_peaks = [(x, y, z, val, tmpl_idx) for (x, y, z, val) in local_peaks]
        partial_peaks[(filename, image_idx, tmpl_idx)] = template_annotated_peaks
        logging.debug(f"[PER-TEMPLATE] {filename}, img={image_idx}, tmpl={tmpl_idx} => {len(local_peaks)} local picks")

    combined_peaks_by_file_img = defaultdict(list)
    for (filename, image_idx, tmpl_idx), peaks in partial_peaks.items():
        combined_peaks_by_file_img[(filename, image_idx)].extend(peaks)

    cc_peaks_dir = os.path.join(output_dir, "cc_peaks")
    os.makedirs(cc_peaks_dir, exist_ok=True)
    aggregated_ccs = defaultdict(list)
    aggregated_avgs = defaultdict(list)
    final_peaks_by_img = {}
    aggregator_npeaks = args.npeaks if args.npeaks else 0
    if args.peak_method == "eman2" and EMAN2_AVAILABLE:
        logging.info(f"Final aggregator: picking up to {aggregator_npeaks} peaks using diameter={args.diameter/2.0} (EMAN2 mode)")
        min_distance = args.diameter / 2.0
    else:
        logging.info(f"Final aggregator: picking up to {aggregator_npeaks} peaks using diameter={args.diameter}")
        min_distance = args.diameter

    for (filename, image_idx), all_candidates in combined_peaks_by_file_img.items():
        logging.info(f"[AGGREGATOR PRE] {filename}, image {image_idx} => {len(all_candidates)} total candidates")
        sorted_candidates = sorted(all_candidates, key=lambda p: p[3], reverse=True)
        final_peaks = final_aggregator_greedy(sorted_candidates, aggregator_npeaks, min_distance)
        logging.info(f"[AGGREGATOR POST] {filename}, image {image_idx} => {len(final_peaks)} final peaks")
        final_peaks_by_img[(filename, image_idx)] = final_peaks
        out_txt = os.path.join(cc_peaks_dir, f"ccs_{filename}_{str(image_idx+1).zfill(2)}.txt")
        with open(out_txt, 'w') as f:
            for (x, y, z, val) in final_peaks:
                f.write(f"{x}\t{y}\t{z}\t{val:.6f}\n")
        logging.info(f"Saved aggregated CC peaks to {out_txt}")
        if args.npeaks and args.npeaks > 0:
            if len(final_peaks) < args.npeaks:
                logging.warning(f"Image {image_idx} in {filename} only has {len(final_peaks)} peaks, not the requested {args.npeaks}")
            aggregated_ccs[filename].extend(final_peaks[:args.npeaks])
        else:
            aggregated_ccs[filename].extend(final_peaks)
        avg_val = np.mean([p[3] for p in final_peaks]) if final_peaks else np.nan
        aggregated_avgs[filename].append((image_idx, avg_val))

    for filename, peaks in aggregated_ccs.items():
        out_file = os.path.join(output_dir, f"ccs_{filename}.txt")
        with open(out_file, 'w') as f:
            for (x, y, z, val) in peaks:
                f.write(f"{x}\t{y}\t{z}\t{val:.6f}\n")
        logging.info(f"Saved aggregated CC peaks for file {filename} to {out_file}")

    for filename, avg_list in aggregated_avgs.items():
        out_file = os.path.join(output_dir, f"ccs_{filename}_avgs.txt")
        with open(out_file, 'w') as f:
            f.write("Image_Index\tAverage_CC\n")
            for (img_idx, avg) in sorted(avg_list, key=lambda x: x[0]):
                f.write(f"{img_idx}\t{avg:.6f}\n")
        logging.info(f"Saved average CC values for file {filename} to {out_file}")

    if len(input_files) == 2:
        data_labels = args.data_labels.split(',')
        file1 = os.path.splitext(os.path.basename(input_files[0]))[0]
        file2 = os.path.splitext(os.path.basename(input_files[1]))[0]
        img_count1 = len(aggregated_avgs[file1])
        img_count2 = len(aggregated_avgs[file2])
        logging.info(f"Processed {img_count1} images from {file1}")
        logging.info(f"Processed {img_count2} images from {file2}")
        avg_data1 = [a for (_, a) in aggregated_avgs[file1]]
        avg_data2 = [a for (_, a) in aggregated_avgs[file2]]
        out_avg_plot = os.path.join(output_dir, "ccs_avgs_plot.png")
        plot_two_distributions(avg_data1, avg_data2, data_labels, out_avg_plot, title="Average CC Value per Image")
        dist_data1 = [p[3] for p in aggregated_ccs[file1]]
        dist_data2 = [p[3] for p in aggregated_ccs[file2]]
        out_dist_plot = os.path.join(output_dir, "ccs_distribution_plot.png")
        plot_two_distributions(dist_data1, dist_data2, data_labels, out_dist_plot, title="All CC Peak Values")
    else:
        file1 = os.path.splitext(os.path.basename(input_files[0]))[0]
        data_label = args.data_labels.split(',')[0]
        img_count1 = len(aggregated_avgs[file1])
        logging.info(f"Processed {img_count1} images from {file1}")
        avg_data = [a for (_, a) in aggregated_avgs[file1]]
        out_avg_plot = os.path.join(output_dir, "ccs_avgs_plot.png")
        plot_two_distributions(avg_data, None, [data_label], out_avg_plot, title="Average CC Value per Image")
        dist_data = [p[3] for p in aggregated_ccs[file1]]
        out_dist_plot = os.path.join(output_dir, "ccs_distribution_plot.png")
        plot_two_distributions(dist_data, None, [data_label], out_dist_plot, title="All CC Peak Values")

    if args.save_cc_maps > 0:
        cc_maps_by_file_img = defaultdict(dict)
        cc_maps_masked_by_file_img = defaultdict(dict)
        for key, value in cc_maps.items():
            if isinstance(value, tuple) and len(value) == 2:
                filename, image_idx, tmpl_idx = key
                cc_map_raw, cc_map_masked = value
                cc_maps_by_file_img[(filename, image_idx)][tmpl_idx] = cc_map_raw
                cc_maps_masked_by_file_img[(filename, image_idx)][tmpl_idx] = cc_map_masked
            else:
                filename, image_idx, tmpl_idx = key
                cc_maps_by_file_img[(filename, image_idx)][tmpl_idx] = value

        for (filename, image_idx), maps_dict in cc_maps_by_file_img.items():
            out_hdf = os.path.join(output_dir, f"{filename}_cc_maps.hdf")
            if EMAN2_AVAILABLE:
                save_cc_maps_stack_eman2(maps_dict, out_hdf)
            else:
                save_cc_maps_stack(maps_dict, out_hdf)
            logging.info(f"Saved {len(maps_dict)} CC maps for file {filename} to {out_hdf}")

        for (filename, image_idx), maps_dict in cc_maps_masked_by_file_img.items():
            if maps_dict:
                out_hdf = os.path.join(output_dir, f"{filename}_cc_maps_masked.hdf")
                if EMAN2_AVAILABLE:
                    save_cc_maps_stack_eman2(maps_dict, out_hdf)
                else:
                    save_cc_maps_stack(maps_dict, out_hdf)
                logging.info(f"Saved {len(maps_dict)} masked CC maps for file {filename} to {out_hdf}")

    # Create coordinate maps for visualization of peaks
    logging.info("Creating coordinate maps with spheres at peak locations")
    coord_maps_by_file = defaultdict(list)
    
    for (filename, image_idx), final_peaks in final_peaks_by_img.items():
        # Only process the first args.save_coords_maps images
        if image_idx >= args.save_coords_maps:
            continue
            
        # Get shape from the original image
        shape = shape_by_file_img[(filename, image_idx)]
        logging.info(f"Creating coordinate map for {filename}, image {image_idx}, shape {shape}, with {len(final_peaks)} peaks")
        
        # Initialize an empty volume
        coords_map = np.zeros(shape, dtype=np.uint8)
        
        # Place a sphere at each peak location
        for i, (x, y, z, val) in enumerate(final_peaks):
            logging.debug(f"  Peak {i+1}: pos=({x}, {y}, {z}), value={val:.4f}, radius={args.coords_map_r}")
            place_sphere(coords_map, (x, y, z), args.coords_map_r)
            
        # Store the map for this file
        coord_maps_by_file[filename].append(coords_map)
        logging.info(f"Created coordinate map for file {filename}, image {image_idx} with {len(final_peaks)} peaks")

    for filename, maps_list in coord_maps_by_file.items():
        maps_to_save = maps_list[:args.save_coords_maps]
        out_map = os.path.join(output_dir, f"{filename}_coords_map.hdf")
        save_coords_map_as_hdf(maps_to_save, out_map)
        logging.info(f"Saved {len(maps_to_save)} coordinate map(s) for file {filename} to {out_map}")

    if args.save_csv:
        csv_out = os.path.join(output_dir, "aggregated_cc_peaks.csv")
        with open(csv_out, 'w', newline='') as csvfile:
            fieldnames = ["Input_File", "Image_Index", "X", "Y", "Z", "CC_Value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for (filename, image_idx), final_peaks in final_peaks_by_img.items():
                for (x, y, z, val) in final_peaks:
                    writer.writerow({
                        "Input_File": filename,
                        "Image_Index": image_idx,
                        "X": x,
                        "Y": y,
                        "Z": z,
                        "CC_Value": f"{val:.6f}"
                    })
        logging.info(f"Saved aggregated CSV summary of all extracted peaks to {csv_out}")

    overall_end = time.time()
    logging.info(f"Total processing time: {overall_end - overall_start:.2f} seconds")
    logging.info(f"Final memory usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Fatal error: " + str(e))
        logging.error(traceback.format_exc())
        sys.exit(1)


'''
FROM CLAUDE; worked April 15 2025
#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
import mrcfile
import h5py
import matplotlib.pyplot as plt
from scipy.signal import correlate, fftconvolve
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import logging
import time
import traceback
import gc  # For explicit garbage collection
import math  # For calculating optimal batch sizes
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import psutil
import csv

try:
    from EMAN2 import EMData, EMNumPy
    EMAN2_AVAILABLE = True
except ImportError:
    EMAN2_AVAILABLE = False

_VERBOSE = False  # Global for controlling debug logs in iterative NMS

#########################
# Utility functions
#########################

def create_output_directory(base_dir):
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "correlation_files"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "coordinate_plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw_coordinates"), exist_ok=True)
    return output_dir

def setup_logging(output_dir, verbose_level):
    log_file = os.path.join(output_dir, "run.log")
    if verbose_level <= 0:
        log_level = logging.ERROR
    elif verbose_level == 1:
        log_level = logging.WARNING
    elif verbose_level == 2:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logging.info("Command: " + " ".join(sys.argv))
    return log_file

def parse_mrc_apix_and_mean(file_path):
    try:
        with mrcfile.open(file_path, permissive=True) as mrc:
            data = mrc.data
            if data is None:
                return (None, None)
            mean_val = data.mean()
            if hasattr(mrc, 'voxel_size') and mrc.voxel_size.x > 0:
                apix = float(mrc.voxel_size.x)
            else:
                nx = mrc.header.nx
                if nx > 0:
                    apix = float(mrc.header.cella.x / nx)
                else:
                    apix = None
            return (apix, mean_val)
    except:
        return (None, None)

def parse_hdf_apix_and_mean(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            apix = None
            mean_val = None
            if 'apix' in f.attrs:
                apix = float(f.attrs['apix'])
            elif 'MDF' in f.keys() and 'apix' in f['MDF'].attrs:
                apix = float(f['MDF'].attrs['apix'])
            if 'MDF/images' in f.keys():
                ds_keys = list(f['MDF/images'].keys())
                if ds_keys:
                    sums = 0.0
                    count = 0
                    for k in ds_keys:
                        if 'image' in f['MDF/images'][k]:
                            arr = f['MDF/images'][k]['image'][:]
                            sums += arr.sum()
                            count += arr.size
                    if count > 0:
                        mean_val = sums / count
            return (apix, mean_val)
    except:
        return (None, None)

    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def check_apix(tomo_path, template_path):
    """
    Checks pixel size (apix) compatibility between template and tomogram.
    Only warns about apix, not intensity differences (which are expected for normalized templates).
    """
    tomo_ext = os.path.splitext(tomo_path)[1].lower()
    template_ext = os.path.splitext(template_path)[1].lower()
    if tomo_ext == '.mrc':
        tomo_apix, _ = parse_mrc_apix_and_mean(tomo_path)
    elif tomo_ext in ['.hdf', '.h5']:
        tomo_apix, _ = parse_hdf_apix_and_mean(tomo_path)
    else:
        tomo_apix = None
    if template_ext == '.mrc':
        tpl_apix, _ = parse_mrc_apix_and_mean(template_path)
    elif template_ext in ['.hdf', '.h5']:
        tpl_apix, _ = parse_hdf_apix_and_mean(template_path)
    else:
        tpl_apix = None
    if tomo_apix and tpl_apix:
        ratio = tpl_apix / tomo_apix if tomo_apix != 0 else None
        if ratio and (ratio < 0.5 or ratio > 2.0):
            logging.warning(f"Template apix {tpl_apix:.2f} vs. tomogram apix {tomo_apix:.2f} differs by >2x.")

def adjust_template(template, target_shape):
    t_shape = template.shape
    new_template = np.zeros(target_shape, dtype=template.dtype)
    slices_target = []
    slices_template = []
    for i, (t_dim, tgt_dim) in enumerate(zip(t_shape, target_shape)):
        if t_dim <= tgt_dim:
            start_tgt = (tgt_dim - t_dim) // 2
            slices_target.append(slice(start_tgt, start_tgt + t_dim))
            slices_template.append(slice(0, t_dim))
        else:
            start_tpl = (t_dim - tgt_dim) // 2
            slices_target.append(slice(0, tgt_dim))
            slices_template.append(slice(start_tpl, start_tpl + tgt_dim))
    new_template[tuple(slices_target)] = template[tuple(slices_template)]
    return new_template

def compute_ncc_map_3d_single(image, template):
    """Compute normalized cross-correlation map of 'image' and 'template' by FFT-based correlation."""
    if image.shape != template.shape:
        template = adjust_template(template, image.shape)
    cc_map = correlate(image, template, mode='same', method='fft')
    # Normalize
    cc_map = (cc_map - np.mean(cc_map)) / np.std(cc_map)
    return cc_map

#########################
# Local NMS methods
#########################

def numpy_to_emdata(a):
    return EMNumPy.numpy2em(a)

# Original reliable non_maximum_suppression function
def non_maximum_suppression(cc_map, n_peaks, diameter):
    """
    This is the original robust NMS algorithm that correctly handles n_peaks.
    Returns list of (x, y, z, value) for the top peaks.
    Guarantees consistent results regardless of input sizes.
    """
    flat = cc_map.ravel()
    sorted_indices = np.argsort(flat)[::-1]
    coords = np.column_stack(np.unravel_index(sorted_indices, cc_map.shape))
    accepted = []
    for idx, coord in enumerate(coords):
        candidate_val = flat[sorted_indices[idx]]
        
        # Skip values that are -inf (could be from thresholding)
        if candidate_val == -np.inf:
            continue
            
        too_close = False
        for (ax, ay, az, aval) in accepted:
            if np.linalg.norm(coord - np.array([ax, ay, az])) < diameter:
                too_close = True
                break
        if not too_close:
            accepted.append((int(coord[0]), int(coord[1]), int(coord[2]), float(candidate_val)))
        if n_peaks > 0 and len(accepted) >= n_peaks:
            break
    return accepted

# Optimized version of NMS for faster performance with large peak counts
def vectorized_non_maximum_suppression(cc_map, n_peaks, diameter):
    """
    Vectorized NMS algorithm that uses numpy optimizations for better performance.
    This function is used only when the user selects 'fallback' method.
    """
    # Get all potential peaks in descending order of value
    flat = cc_map.ravel()
    
    # For efficiency, limit initial sorting to the top 10x n_peaks values
    # This prevents O(n log n) sorting of potentially millions of points
    if n_peaks > 0:
        # Efficiently find top candidates - much faster for large arrays
        min_candidates = min(10 * n_peaks, flat.size)
        top_indices = np.argpartition(flat, -min_candidates)[-min_candidates:]
        # Sort just these candidates
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
    else:
        # If n_peaks is 0, we sort all values (this could be very slow for large arrays)
        top_indices = np.argsort(flat)[::-1]
    
    # Convert flat indices to 3D coordinates
    coords = np.asarray(np.unravel_index(top_indices, cc_map.shape)).T
    
    accepted = []
    # Squared diameter for faster distance calculation
    diameter_squared = diameter * diameter
    
    # Use numpy vectors for faster distance calculations
    for idx, coord in enumerate(coords):
        # Skip early if we've found enough peaks
        if n_peaks > 0 and len(accepted) >= n_peaks:
            break
            
        candidate_val = flat[top_indices[idx]]
        
        # Skip values that are -inf (could be from thresholding)
        if candidate_val == -np.inf:
            continue
            
        # Check if this peak is too close to any accepted peak
        too_close = False
        if accepted:
            # Convert accepted peaks to numpy array for vectorized distance calc
            accepted_coords = np.array([[x, y, z] for (x, y, z, _) in accepted])
            
            # Compute squared distances to all accepted peaks at once
            # This is much faster than looping through each accepted peak
            diff = accepted_coords - coord
            sq_distances = np.sum(diff * diff, axis=1)
            
            # If any distance is less than diameter, it's too close
            if np.any(sq_distances < diameter_squared):
                too_close = True
                
        if not too_close:
            accepted.append((int(coord[0]), int(coord[1]), int(coord[2]), float(candidate_val)))
    
    return accepted

def mask_out_region_xyz(arr, x0, y0, z0, diameter):
    """
    Sets arr[z, y, x] = -inf for all points within 'diameter' of (x0,y0,z0).
    
    Parameters:
    - arr: NumPy array with shape (nz, ny, nx)
    - x0, y0, z0: coordinates of the center point in (x,y,z) format
    - diameter: diameter of the sphere to mask
    
    Note: This function properly handles the coordinate conversion from
    input (x,y,z) coordinates to NumPy's [z,y,x] indexing.
    """
    nz, ny, nx = arr.shape
    dd = diameter**2
    radius = int(diameter)
    
    # Loop using z,y,x iteration order for better cache locality
    for z in range(max(0, z0 - radius), min(nz, z0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(ny, y0 + radius + 1)):
            for x in range(max(0, x0 - radius), min(nx, x0 + radius + 1)):
                dx = x - x0
                dy = y - y0
                dz = z - z0
                if (dx*dx + dy*dy + dz*dz) <= dd:
                    # Use NumPy [z,y,x] indexing
                    arr[z, y, x] = -np.inf
    return arr

def iterative_nms_eman2(cc_map_np, n_peaks, diameter, cc_thresh, verbose=False):
    """
    Modified EMAN2-based NMS with proper n_peaks handling.
    Uses radius (half of diameter) for masking spheres around detected peaks.
    Properly handles coordinate system conversion between EMAN2 and NumPy.
    
    IMPORTANT: EMAN2 uses (x,y,z) coordinates but NumPy arrays are indexed as [z,y,x]
    """
    from EMAN2 import EMNumPy, EMData
    
    # Ensure diameter is not None
    if diameter is None:
        diameter = 35.0  # Default if not provided
        
    # Calculate radius as half of the diameter
    radius = diameter / 2.0
    radius_squared = radius * radius
    
    if verbose:
        print(f"[DEBUG] iterative_nms_eman2 start, n_peaks={n_peaks}, diameter={diameter:.1f}, radius={radius:.1f}")

    # Work with a copy of the input array
    arr = cc_map_np.copy().astype(np.float32)
    shape_nz, shape_ny, shape_nx = arr.shape  # NumPy shape is (z,y,x)

    # Apply threshold if requested
    if cc_thresh > 0:
        val_thr = arr.mean() + cc_thresh * arr.std()
        arr = np.where(arr >= val_thr, arr, -np.inf)

    # Prepare to pick peaks
    if (not n_peaks) or (n_peaks <= 0):
        n_peaks = arr.size

    accepted = []
    
    # Main peak finding loop
    for i in range(n_peaks):
        # Convert to EMData only for finding the maximum location
        # This is what EMAN2 is good at doing quickly
        em = EMNumPy.numpy2em(arr)
        max_val = em["maximum"]
        
        # Stop if no more valid peaks
        if max_val == -np.inf:
            break

        # Find the location of the maximum value in EMAN2 coordinates (x,y,z)
        eman_x, eman_y, eman_z = em.calc_max_location()
        eman_x, eman_y, eman_z = int(eman_x), int(eman_y), int(eman_z)
        
        # Back to NumPy for the rest of processing
        del em  # Free memory

        # IMPORTANT: Convert EMAN2 coordinates (x,y,z) to NumPy array indices [z,y,x]
        # For array access, we need to use numpy_z = eman_z, numpy_y = eman_y, numpy_x = eman_x
        
        if not (0 <= eman_z < shape_nz and 0 <= eman_y < shape_ny and 0 <= eman_x < shape_nx):
            if verbose:
                print(f"[WARNING] EMAN2 gave out-of-bounds peak (x={eman_x},y={eman_y},z={eman_z}), ignoring it.")
        else:
            # Store the peak in the format (x,y,z,val) as expected by the rest of the code
            accepted.append((eman_x, eman_y, eman_z, float(max_val)))
            if verbose:
                print(f"[DEBUG] EMAN2 peak {i+1}: (x={eman_x}, y={eman_y}, z={eman_z}), val={max_val:.3f}")

        # Explicitly mask out a spherical region around the peak
        # NOTE: We need to use NumPy indexing [z,y,x] while the peak is in EMAN2 (x,y,z) format
        z_min = max(0, eman_z - int(radius))
        z_max = min(shape_nz, eman_z + int(radius) + 1)
        y_min = max(0, eman_y - int(radius))
        y_max = min(shape_ny, eman_y + int(radius) + 1)
        x_min = max(0, eman_x - int(radius))
        x_max = min(shape_nx, eman_x + int(radius) + 1)
        
        # Mask the spherical region in NumPy's [z,y,x] coordinate system
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # Check if this point is within the sphere
                    # The distance calculation needs to match EMAN2 coordinates
                    if ((x - eman_x)**2 + (y - eman_y)**2 + (z - eman_z)**2) <= radius_squared:
                        # Access using NumPy [z,y,x] indexing
                        arr[z, y, x] = -np.inf
        
        if verbose and i < 3:  # Only for the first few iterations to avoid excessive output
            # Check if masking was effective - use NumPy [z,y,x] indexing!
            if arr[eman_z, eman_y, eman_x] != -np.inf:
                print(f"[ERROR] Masking failed! Value at peak location after masking: {arr[eman_z, eman_y, eman_x]}")
            else:
                print(f"[DEBUG] Masking successful at iteration {i+1}")

    return accepted, arr

def fallback_iterative_nms(cc_map, n_peaks, diameter, cc_thresh, verbose=False):
    """
    Non-EMAN2 version of iterative NMS. Used when EMAN2 is not available.
    
    This function works with NumPy arrays in [z,y,x] indexing format, but returns
    peak coordinates in (x,y,z,val) format for compatibility with the rest of the code.
    
    Parameters:
    - cc_map: 3D NumPy array of cross-correlation values, shape (z,y,x)
    - n_peaks: Number of peaks to find (0 means find all)
    - diameter: Minimum distance between peaks
    - cc_thresh: Threshold in sigmas above mean (0 means no threshold)
    - verbose: Whether to print debug information
    
    Returns:
    - accepted: List of (x,y,z,val) peaks
    - arr: masked correlation map
    """
    arr = cc_map.copy()
    nz, ny, nx = arr.shape

    # threshold if needed
    if cc_thresh > 0:
        val_thr = arr.mean() + cc_thresh * arr.std()
        arr = np.where(arr >= val_thr, arr, -np.inf)

    if (not n_peaks) or (n_peaks <= 0):
        n_peaks = arr.size

    accepted = []
    # Use the simple, reliable algorithm
    for i in range(n_peaks):
        max_val = arr.max()
        if max_val == -np.inf:
            break
            
        max_idx = np.argmax(arr)
        # NumPy unravel_index gives (z,y,x) for a 3D array with shape (nz,ny,nx)
        z0, y0, x0 = np.unravel_index(max_idx, arr.shape)

        # Store as (x,y,z,val) as expected by the aggregator
        accepted.append((x0, y0, z0, float(max_val)))
        if verbose:
            print(f"[DEBUG fallback] peak {i+1}: (x={x0}, y={y0}, z={z0}), val={max_val:.3f}")

        # mask_out_region_xyz handles the coordinate conversion correctly
        arr = mask_out_region_xyz(arr, x0, y0, z0, diameter)

    return accepted, arr

#########################
# Output saving functions
#########################

def plot_3d_peak_coordinates(output_dir, filename, template_str, image_idx, peak_coords):
    if np.asarray(peak_coords).size == 0:
        return
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peak_coords[:,0], peak_coords[:,1], peak_coords[:,2], marker='o')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Top CC Peak Locations")
    image_str = str(image_idx).zfill(3)
    out_png = os.path.join(output_dir, 'coordinate_plots', f'cc_{filename}_T{template_str}_I{image_str}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    logging.info(f"Saved 3D peak plot to {out_png}")

def place_sphere(volume, center, radius):
    """
    Places a sphere in a 3D volume by setting voxels to 1.
    - volume: 3D NumPy array with shape (z,y,x)
    - center: (x,y,z) coordinates of sphere center
    - radius: radius of the sphere
    
    Note: This handles the coordinate conversion from (x,y,z) -> [z,y,x]
    """
    x0, y0, z0 = center
    nz, ny, nx = volume.shape  # NumPy array has shape (z,y,x)
    r2 = radius**2
    
    # Convert x,y,z coords to NumPy's [z,y,x] indexing
    for z in range(z0 - radius, z0 + radius + 1):
        if z < 0 or z >= nz:
            continue
        for y in range(y0 - radius, y0 + radius + 1):
            if y < 0 or y >= ny:
                continue
            for x in range(x0 - radius, x0 + radius + 1):
                if x < 0 or x >= nx:
                    continue
                if (x - x0)**2 + (y - y0)**2 + (z - z0)**2 <= r2:
                    # Access using NumPy [z,y,x] indexing
                    volume[z, y, x] = 1

def save_coords_map_as_hdf(coords_map_stack, out_path):
    """
    Save coordinate maps as HDF stack in a format compatible with EMAN2.
    Creates a structure with MDF/images/i/image datasets.
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    with h5py.File(out_path, 'w') as f:
        # Add attributes needed for EMAN2 compatibility
        f.attrs['appion_version'] = np.string_('ccpeaks_plot')
        f.attrs['MDF'] = np.string_('1')  # This is needed for EMAN2 compatibility
        
        # Create group structure
        mdf = f.create_group('MDF')
        mdf.attrs['version'] = np.string_('1.0')
        images = mdf.create_group('images')
        
        # Add each image to the stack
        for i in range(len(coords_map_stack)):
            img_group = images.create_group(str(i))
            img_group.create_dataset('image', data=coords_map_stack[i], dtype=coords_map_stack[i].dtype)
    
    logging.info(f"Saved EMAN2-compatible coords_map to {out_path}")

def save_cc_maps_stack_eman2(cc_maps_dict, out_path):
    """
    Use EMAN2 to save multiple CC maps in a single .hdf stack.
    Each map is written as one slice.
    """
    if not EMAN2_AVAILABLE:
        logging.error("EMAN2 not available; cannot save cc maps in EMAN2 format.")
        return
    
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    sorted_keys = sorted(cc_maps_dict.keys())
    for i, tmpl_idx in enumerate(sorted_keys):
        arr = cc_maps_dict[tmpl_idx]
        em = EMNumPy.numpy2em(arr)
        # Write directly using EMAN2's native format
        em.write_image(out_path, i)
    logging.info(f"Saved {len(sorted_keys)} CC map(s) to {out_path} using EMAN2")

def save_cc_maps_stack(cc_maps_dict, out_path):
    """
    HDF5-based alternative when EMAN2 is not available.
    Creates a stack in EMAN2-compatible format.
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    sorted_keys = sorted(cc_maps_dict.keys())
    with h5py.File(out_path, 'w') as f:
        # Add attributes needed for EMAN2 compatibility
        f.attrs['appion_version'] = np.string_('ccpeaks_plot')
        f.attrs['MDF'] = np.string_('1')
        
        # Create group structure
        mdf = f.create_group('MDF')
        mdf.attrs['version'] = np.string_('1.0')
        images = mdf.create_group('images')
        
        # Add each image to the stack
        for i, tmpl_idx in enumerate(sorted_keys):
            img_group = images.create_group(str(i))
            img_group.create_dataset('image', data=cc_maps_dict[tmpl_idx], dtype=cc_maps_dict[tmpl_idx].dtype)
    
    logging.info(f"Saved {len(sorted_keys)} CC map(s) to {out_path} using HDF5")

#########################
# compute_auto_diameter
#########################

def compute_auto_diameter(template, tomo_shape):
    """
    Compute a more appropriate diameter for non-spherical objects by considering
    both the longest and shortest spans of the template.
    
    For non-spherical objects, using just the longest span can be too conservative.
    Using the average of the longest and shortest spans provides a better estimate
    for the minimum distance between densely packed objects.
    
    Parameters:
        template: 3D volume
        tomo_shape: Shape of the tomogram
        
    Returns:
        Average of the longest and shortest spans, capped by the tomogram dimension
    """
    # Smooth the template to reduce noise
    smoothed = gaussian_filter(template, sigma=1)
    
    # Threshold to identify meaningful density
    thresh = smoothed.mean() + 2 * smoothed.std()
    binary = (smoothed >= thresh).astype(np.uint8)
    
    # Find coordinates of all non-zero points
    indices = np.argwhere(binary)
    if indices.size == 0:
        return 0
    
    # Calculate spans in all dimensions
    min_indices = indices.min(axis=0)
    max_indices = indices.max(axis=0)
    spans = max_indices - min_indices + 1  # Add 1 to include both ends
    
    # Get longest and shortest spans
    longest_span = spans.max()
    shortest_span = spans.min()
    
    # Use average of longest and shortest spans
    average_span = (longest_span + shortest_span) / 2.0
    
    # Cap by the smallest tomogram dimension
    smallest_tomo_dim = min(tomo_shape)
    auto_diameter = min(average_span, smallest_tomo_dim)
    
    logging.info(f"Auto-diameter calculation: longest_span={longest_span}, shortest_span={shortest_span}, average={average_span:.1f}, final={auto_diameter:.1f}")
    
    return auto_diameter

# Removed duplicate process_image_template function

#########################
# Final aggregator
#########################

def final_aggregator_greedy(candidate_list, n_peaks, min_distance):
    """
    Sort candidates by CC value descending, pick each if
    it's at least 'min_distance' away from all previously accepted.
    Returns up to n_peaks.

    candidate_list: list of (x, y, z, val) or (x, y, z, val, template_idx)
    n_peaks: desired max number of final picks
    min_distance: float, minimum allowed distance between peaks
    
    Returns: List of (x, y, z, val) tuples - template_idx is removed if present
    """
    # Sort descending by CC value (always 4th element, index 3)
    candidate_list.sort(key=lambda x: x[3], reverse=True)
    accepted = []

    # Use the same consistent NMS approach as before
    for cand in candidate_list:
        # Get coordinates and value, handling both formats
        # Original format: (x, y, z, val)
        # Debug format with template: (x, y, z, val, template_idx)
        if len(cand) >= 5:
            x, y, z, val, tmpl_idx = cand[:5]  # Unpack with template index
        else:
            x, y, z, val = cand[:4]  # Unpack without template index
            tmpl_idx = None
            
        too_close = False

        # Check distance to all previously accepted peaks
        for accepted_peak in accepted:
            ax, ay, az = accepted_peak[:3]  # Get coordinates of accepted peak
            a_val = accepted_peak[3]  # Get value of accepted peak
            
            dist = np.linalg.norm([x - ax, y - ay, z - az])
            if dist < min_distance:
                tmpl_str = f", template={tmpl_idx}" if tmpl_idx is not None else ""
                logging.debug(
                    f"[DEBUG aggregator] EXCLUDING peak "
                    f"(x={x}, y={y}, z={z}, val={val:.3f}{tmpl_str}) "
                    f"because it's too close to "
                    f"(x={ax}, y={ay}, z={az}, val={a_val:.3f}) "
                    f"[dist={dist:.3f}, min_distance={min_distance}]"
                )
                too_close = True
                break

        if not too_close:
            # Always store as (x, y, z, val) format without template info
            accepted.append((x, y, z, val))
            tmpl_str = f", template={tmpl_idx}" if tmpl_idx is not None else ""
            logging.debug(
                f"[DEBUG aggregator] ACCEPTING peak #{len(accepted)} => "
                f"(x={x}, y={y}, z={z}, val={val:.3f}{tmpl_str})"
            )

        # Stop if we've reached the requested number of peaks
        if n_peaks > 0 and len(accepted) >= n_peaks:
            break

    return accepted

#########################
# Plot distribution
#########################

def plot_two_distributions(data1, data2, labels, output_path, title="CC Distribution"):
    """
    Create violin plots comparing one or two distributions of cross-correlation values.
    
    Parameters:
    - data1: First dataset (required)
    - data2: Second dataset (optional)
    - labels: Labels for the datasets
    - output_path: Where to save the plot
    - title: Main title for the plot
    """
    # Handle empty data
    if len(data1) == 0:
        logging.warning(f"plot_two_distributions: data1 is empty, cannot create plot {output_path}")
        return
    data1 = np.asarray(data1)
    if data2 is not None and len(data2) > 0:
        data2 = np.asarray(data2)
    else:
        data2 = None

    if data2 is None or len(data2) == 0:
        # Single distribution
        fig, ax = plt.subplots(figsize=(8,8))
        parts = ax.violinplot([data1], positions=[1], showmeans=False, showmedians=False, showextrema=False)
        ax.scatter(np.full(data1.shape, 1), data1)
        q1, med, q3 = np.percentile(data1, [25,50,75])
        mean_val = np.mean(data1)
        ax.scatter(1, med, marker='s', color='red', zorder=3)
        ax.scatter(1, mean_val, marker='o', color='white', edgecolor='black', zorder=3)
        ax.vlines(1, data1.min(), data1.max(), color='black', lw=2)
        ax.vlines(1, q1, q3, color='black', lw=5)
        ax.set_xticks([1])
        ax.set_xticklabels(labels[:1])
        ax.set_ylabel("CC Values")
        ax.set_title(f"{title}\nN={len(data1)}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved single-distribution plot to {output_path}")
    else:
        # Two distributions
        def significance_from_pval(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            else: return "ns"
        def is_normal(dat):
            if len(dat) < 3:
                return False
            stat, p = shapiro(dat)
            return p > 0.05
        def calculate_effect_size(dat1, dat2):
            normal1 = is_normal(dat1)
            normal2 = is_normal(dat2)
            if normal1 and normal2:
                tstat, pval = ttest_ind(dat1, dat2, equal_var=False)
                mean1, mean2 = np.mean(dat1), np.mean(dat2)
                var1, var2 = np.var(dat1, ddof=1), np.var(dat2, ddof=1)
                n1, n2 = len(dat1), len(dat2)
                pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                effect = 0.0 if pooled_std < 1e-12 else (mean1 - mean2) / pooled_std
                return pval, effect, "Cohen's d"
            else:
                stat, pval = mannwhitneyu(dat1, dat2, alternative='two-sided')
                n1, n2 = len(dat1), len(dat2)
                effect = 1 - (2*stat)/(n1*n2)  # rank-biserial
                return pval, effect, "Rank-biserial"
        pval, effect, method = calculate_effect_size(data1, data2)
        sig = significance_from_pval(pval)
        N1, N2 = len(data1), len(data2)

        fig, ax = plt.subplots(figsize=(8,8))
        parts = ax.violinplot([data1, data2], positions=[1,2], showmeans=False, showmedians=False, showextrema=False)
        colors = ['#1f77b4', '#ff7f0e']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.3)

        # Distribution 1
        ax.scatter(np.full(data1.shape, 1), data1, edgecolor='k')
        q1_1, med1, q3_1 = np.percentile(data1, [25,50,75])
        mean1 = np.mean(data1)
        ax.scatter(1, med1, marker='s', color='red', zorder=3)
        ax.scatter(1, mean1, marker='o', color='white', edgecolor='black', zorder=3)
        ax.vlines(1, data1.min(), data1.max(), color='black', lw=2)
        ax.vlines(1, q1_1, q3_1, color='black', lw=5)

        # Distribution 2
        ax.scatter(np.full(data2.shape, 2), data2, edgecolor='k')
        q1_2, med2, q3_2 = np.percentile(data2, [25,50,75])
        mean2 = np.mean(data2)
        ax.scatter(2, med2, marker='s', color='red', zorder=3)
        ax.scatter(2, mean2, marker='o', color='white', edgecolor='black', zorder=3)
        ax.vlines(2, data2.min(), data2.max(), color='black', lw=2)
        ax.vlines(2, q1_2, q3_2, color='black', lw=5)

        ax.set_xticks([1,2])
        ax.set_xticklabels(labels)
        ax.set_ylabel("CC Values")
        
        # Enhanced title with more information
        ax.set_title(f"{title}\n{labels[0]}: {N1} values, {labels[1]}: {N2} values\n"
                    f"p={pval:.6f} ({sig}), {method}={effect:.4f}\n"
                    f"Mean: {mean1:.3f} vs {mean2:.3f}, Median: {med1:.3f} vs {med2:.3f}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved two-distribution plot to {output_path}")

#########################
# Main function
#########################

def main():
    from collections import defaultdict
    parser = argparse.ArgumentParser(
        description="Compute 3D cross-correlation maps between tomograms and templates, then extract and analyze peaks.\n"
        "Process all images in input files but saves coordinate maps only for the first --save_coords_map images to save space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single tomogram and single template:
  python ccpeaks_plot_claude.py --input tomogram.hdf --template template.hdf --npeaks 20 --diameter 13

  # Compare two tomogram files against many template rotations:
  python ccpeaks_plot_claude.py --input tomo1.hdf,tomo2.hdf --template templates_rotated.hdf --npeaks 36 --diameter 15 --threads 32 --peak_method eman2
  
  # Process all images but only save coordinate maps for first 10 images:
  python ccpeaks_plot_claude.py --input tomo_stack.hdf --template template.hdf --save_coords_map 10 --npeaks 50 --diameter 13
""")

    # Parameters (alphabetically ordered)
    parser.add_argument('--cc_thresh', dest='ccc_thresh_sigma', type=float, default=0.0,
                        help="Threshold for picking peaks (in std above mean). Default=0.0 (no threshold).")
    parser.add_argument('--coords_map_r', type=int, default=3,
                        help="Radius of spheres in coordinate maps. Default=3.")
    parser.add_argument('--data_labels', type=str, default="file1,file2",
                        help="Comma-separated labels for the datasets. Default='file1,file2'.")
    parser.add_argument('--diameter', type=float, default=None,
                        help="Minimum distance in voxels between final peaks. Default=None (auto-computed).")
    parser.add_argument('--input', required=True,
                        help="Comma-separated input file paths (tomograms). Maximum 2 files allowed.")
    parser.add_argument('--npeaks', type=int, default=None,
                        help="Number of final peaks to keep per image. Default=None (auto-calculated).")
    parser.add_argument('--output_dir', default="cc_analysis",
                        help="Base name for output directory. Default='cc_analysis'.")
    parser.add_argument('--peak_method', type=str, default="original",
                        help="Peak finding method (options: 'original', 'eman2', 'fallback'). Default='original'.")
    parser.add_argument('--save_cc_maps', type=int, default=1,
                        help="How many CC maps to save per input file. Default=1.")
    parser.add_argument('--save_coords_maps', type=int, default=1,
                        help="How many coordinate maps to save per input file. Default=1.")
    parser.add_argument('--save_csv', action='store_true', default=False,
                        help="Export a CSV summary of all extracted peaks. Default=False.")
    parser.add_argument('--template', required=True,
                        help="Template image file path (can be an HDF stack).")
    parser.add_argument('--threads', type=int, default=1,
                        help="Number of processes to use in parallel. Default=1.")
    parser.add_argument('--verbose', type=int, default=2,
                        help="Verbosity level (0=ERROR,1=WARNING,2=INFO,3=DEBUG). Default=2.")

    args = parser.parse_args()

    # Basic validations
    if args.save_cc_maps < 1:
        logging.error("--save_cc_maps must be >= 1.")
        sys.exit(1)
        
    if args.save_coords_maps < 1:
        logging.error("--save_coords_maps must be >= 1.")
        sys.exit(1)
    
    # Check number of input files
    input_files = args.input.split(',')
    if len(input_files) > 2:
        logging.error("Maximum of 2 input files allowed. Please provide at most two files separated by commas.")
        sys.exit(1)

    global _VERBOSE
    _VERBOSE = (args.verbose >= 3)

    output_dir = create_output_directory(args.output_dir)
    setup_logging(output_dir, args.verbose)

    overall_start = time.time()
    logging.info("Starting processing...")

    # Log EMAN2 availability
    logging.info(f"EMAN2 is {'available' if EMAN2_AVAILABLE else 'not available'}")
    
    # Load template(s)
    templates_list = load_image(args.template)
    if (templates_list is None) or (len(templates_list) == 0):
        logging.error("Could not load template file. Exiting.")
        sys.exit(1)
    templates = [t for t in templates_list]
    n_templates = len(templates)
    logging.info(f"Loaded {n_templates} template(s) from {args.template}")

    # Enforce that save_ccmaps is not > number of templates
    if args.save_cc_maps > n_templates:
        logging.info(f"--save_ccmaps={args.save_cc_maps} but only {n_templates} template(s) present. Reducing to {n_templates}.")
        args.save_cc_maps = n_templates

    # Load input files and check sizes
    image_counts = []
    input_images = []
    
    for file_path in input_files:
        image_list = load_image(file_path)
        if (image_list is None) or (len(image_list) == 0):
            logging.error(f"Could not load tomogram {file_path}. Exiting.")
            sys.exit(1)
        image_counts.append(len(image_list))
        input_images.append(image_list)
    
    # Get shape from first image of first file
    first_tomo_shape = input_images[0][0].shape
    
    # Cap the save_coords_map to minimum count of images
    min_images = min(image_counts)
    if args.save_coords_maps > min_images:
        logging.info(f"--save_coords_map={args.save_coords_maps} but minimum image count is {min_images}. Reducing to {min_images}.")
        args.save_coords_maps = min_images

    # Auto diameter if needed
    if args.diameter is None:
        auto_diameter = compute_auto_diameter(templates[0], first_tomo_shape)
        args.diameter = auto_diameter
        logging.info(f"No diameter provided; using auto-derived diameter: {auto_diameter}")

    process_mem = psutil.Process(os.getpid())
    logging.info(f"Starting memory usage: {process_mem.memory_info().rss / (1024 * 1024):.2f} MB")

    tasks = []
    shape_by_file_img = {}
    for i, file_path in enumerate(input_files):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        check_apix(file_path, args.template)  # Use the renamed function (no intensity check)
        
        # Use the already loaded images to avoid loading them again
        image_list = input_images[i]
        
        # Process ALL images for peak finding, not just the first save_coords_map
        for img_idx, image_3d in enumerate(image_list):
            shape_by_file_img[(filename, img_idx)] = image_3d.shape
            
            # Track which images should have coordinate maps saved (limited by save_coords_map)
            should_save_coords_map = img_idx < args.save_coords_maps

            # Calculate local_npeaks (how many peaks to find per template)
            if args.npeaks is None:
                diam_int = int(args.diameter)
                n_possible = (image_3d.shape[0] // diam_int) * (image_3d.shape[1] // diam_int) * (image_3d.shape[2] // diam_int)
                local_npeaks = n_possible
            else:
                local_npeaks = args.npeaks

            # For each template
            for tmpl_idx, template_data in enumerate(templates):
                store_map = (tmpl_idx < args.save_cc_maps)  # only store CC map for first N templates
                tasks.append((
                    filename,
                    img_idx,
                    tmpl_idx,
                    image_3d,
                    template_data,
                    local_npeaks,         # local picks
                    args.ccc_thresh_sigma,
                    args.diameter,
                    store_map,
                    args.peak_method,
                    args.template
                ))

    # Process all tasks
    partial_results = []
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_image_template, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
            res = fut.result()
            if res is not None:
                partial_results.append(res)

    partial_peaks = defaultdict(list)
    cc_maps = {}

    # Collect local picks
    for res in partial_results:
        if len(res) == 6:  # EMAN2 debug format with both original and masked CC maps
            filename, image_idx, tmpl_idx, local_peaks, cc_map_raw, cc_map_masked = res
            cc_maps[(filename, image_idx, tmpl_idx)] = (cc_map_raw, cc_map_masked)  # Store as tuple
            logging.info(f"Saved both original and masked CC maps for debugging (template {tmpl_idx})")
        elif len(res) == 5:  # With regular CC map
            filename, image_idx, tmpl_idx, local_peaks, cc_map_raw = res
            cc_maps[(filename, image_idx, tmpl_idx)] = cc_map_raw
        else:  # Without CC map
            filename, image_idx, tmpl_idx, local_peaks = res

        # Store template index with each peak for debugging
        # Add the template index as a 5th element in each peak tuple
        template_annotated_peaks = [(x, y, z, val, tmpl_idx) for (x, y, z, val) in local_peaks]
        partial_peaks[(filename, image_idx, tmpl_idx)] = template_annotated_peaks
        logging.debug(f"[PER-TEMPLATE] {filename}, img={image_idx}, tmpl={tmpl_idx} => {len(local_peaks)} local picks")

    # Combine local picks (all templates) for each (filename, image_idx)
    combined_peaks_by_file_img = defaultdict(list)
    for (filename, image_idx, tmpl_idx), peaks in partial_peaks.items():
        combined_peaks_by_file_img[(filename, image_idx)].extend(peaks)

    # Now do a final aggregator that picks exactly args.npeaks (if possible) out of the combined list
    cc_peaks_dir = os.path.join(output_dir, "cc_peaks")
    os.makedirs(cc_peaks_dir, exist_ok=True)

    aggregated_ccs = defaultdict(list)
    aggregated_avgs = defaultdict(list)
    final_peaks_by_img = {}

    # If user didn't specify n_peaks, we do the same auto-calc used for local
    aggregator_npeaks = args.npeaks if args.npeaks else 0
    
    # When using EMAN2 method, use half-diameter for consistency
    # since we already corrected EMAN2 to use radius = diameter/2
    if args.peak_method == "eman2" and EMAN2_AVAILABLE:
        # For EMAN2, use half-diameter to be consistent with our EMAN2 fix
        logging.info(f"Final aggregator: picking up to {aggregator_npeaks} peaks from combined candidates, using method={args.peak_method} with diameter={args.diameter}...")
    else:
        # For other methods, use the regular diameter value
        logging.info(f"Final aggregator: picking up to {aggregator_npeaks} peaks from combined candidates, using method={args.peak_method} with diameter={args.diameter}...")

    for (filename, image_idx), all_candidates in combined_peaks_by_file_img.items():
        logging.info(f"[AGGREGATOR PRE] {filename}, image={image_idx} => {len(all_candidates)} total candidates from all templates")

        # Dump the top few candidate peaks for debugging purposes
        logging.info(f"DEBUG - CANDIDATE PEAKS before aggregation for {filename}, image {image_idx}:")
        logging.info(f"NOTE: These candidates come from ALL templates combined! They may be close to each other because")
        logging.info(f"      they were found in different template correlations. The final peaks after aggregation will")
        logging.info(f"      enforce the minimum distance of {args.diameter} pixels between peaks.")
        
        # Sort by value (4th element)
        sorted_candidates = sorted(all_candidates, key=lambda p: p[3], reverse=True)
        max_to_show = min(10, len(sorted_candidates))  # Just show top 10 for debugging
        for i, peak in enumerate(sorted_candidates[:max_to_show]):
            # Handle peaks with or without template index 
            if len(peak) >= 5:
                x, y, z, val, tmpl_idx = peak
                logging.info(f"  Candidate #{i+1}: (x={x}, y={y}, z={z}), val={val:.3f}, template={tmpl_idx}")
            else:
                x, y, z, val = peak
                logging.info(f"  Candidate #{i+1}: (x={x}, y={y}, z={z}), val={val:.3f}")
                
        if len(sorted_candidates) > max_to_show:
            logging.info(f"  ... and {len(sorted_candidates) - max_to_show} more candidates (total: {len(sorted_candidates)})")
        
        # When using EMAN2 method, the mask.sharp is using radius (diameter/2) 
        # so for consistency, we need to use the same minimum distance in the final aggregator
        if args.peak_method == "eman2" and EMAN2_AVAILABLE:
            # For EMAN2, we need to use diameter/2 because that's what the EMAN2 masking uses
            min_distance = args.diameter / 2.0
            logging.info(f"Using aggregator with EMAN2-compatible distance={min_distance} (half of diameter {args.diameter})")
        else:
            # For other methods, use the original diameter value
            min_distance = args.diameter
            logging.info(f"Using aggregator with original distance={min_distance}")
            
        final_peaks = final_aggregator_greedy(all_candidates, aggregator_npeaks, min_distance)

        # Dump the final peaks for debugging
        logging.info(f"FINAL PEAKS after aggregation for {filename}, image {image_idx}:")
        for i, (x, y, z, val) in enumerate(final_peaks):
            logging.info(f"  Final Peak #{i+1}: (x={x}, y={y}, z={z}), val={val:.3f}")

        logging.info(f"[AGGREGATOR POST] {filename}, image={image_idx} => {len(final_peaks)} final peaks")

        final_peaks_by_img[(filename, image_idx)] = final_peaks
        # Write out to a text file
        out_txt = os.path.join(cc_peaks_dir, f"ccs_{filename}_{str(image_idx+1).zfill(2)}.txt")
        with open(out_txt, 'w') as f:
            for (x, y, z, val) in final_peaks:
                f.write(f"{x}\t{y}\t{z}\t{val:.6f}\n")
        logging.info(f"Saved aggregated CC peaks to {out_txt}")

        # Accumulate for distribution
        # Ensure we use exactly npeaks from each image for statistical consistency
        # If npeaks was specified, use that many peaks, otherwise use all available peaks
        if args.npeaks and args.npeaks > 0:
            # Use exactly npeaks per image (truncate or pad with NaN if necessary) 
            if len(final_peaks) >= args.npeaks:
                # If we have enough or more peaks, use exactly npeaks
                aggregated_ccs[filename].extend(final_peaks[:args.npeaks])
            else:
                # If we don't have enough peaks, log a warning
                logging.warning(f"Image {image_idx} in {filename} only has {len(final_peaks)} peaks, not the requested {args.npeaks}")
                aggregated_ccs[filename].extend(final_peaks)
        else:
            # If npeaks wasn't specified, use all available peaks
            aggregated_ccs[filename].extend(final_peaks)
            
        # Calculate average CC value for this image
        avg_val = np.mean([p[3] for p in final_peaks]) if final_peaks else np.nan
        aggregated_avgs[filename].append((image_idx, avg_val))

    # Write global aggregated results
    for filename, peaks in aggregated_ccs.items():
        out_file = os.path.join(output_dir, f"ccs_{filename}.txt")
        with open(out_file, 'w') as f:
            for (x, y, z, val) in peaks:
                f.write(f"{x}\t{y}\t{z}\t{val:.6f}\n")
        logging.info(f"Saved aggregated CC peaks for file {filename} to {out_file}")

    # Write average CC
    for filename, avg_list in aggregated_avgs.items():
        out_file = os.path.join(output_dir, f"ccs_{filename}_avgs.txt")
        with open(out_file, 'w') as f:
            f.write("Image_Index\tAverage_CC\n")
            for (img_idx, av) in sorted(avg_list, key=lambda x: x[0]):
                f.write(f"{img_idx}\t{av:.6f}\n")
        logging.info(f"Saved average CC values for file {filename} to {out_file}")

    # Plot and save distribution statistics
    if len(input_files) == 2:
        data_labels = args.data_labels.split(',')
        file1 = os.path.splitext(os.path.basename(input_files[0]))[0]
        file2 = os.path.splitext(os.path.basename(input_files[1]))[0]
        
        # Get the total number of images processed for each file
        img_count1 = len(aggregated_avgs[file1])
        img_count2 = len(aggregated_avgs[file2])
        
        # Log number of images processed
        logging.info(f"Processed {img_count1} images from {file1}")
        logging.info(f"Processed {img_count2} images from {file2}")
        
        # Average CC values per image
        avg_data1 = [a for (_, a) in aggregated_avgs[file1]]
        avg_data2 = [a for (_, a) in aggregated_avgs[file2]]
        out_avg_plot = os.path.join(output_dir, "ccs_avgs_plot.png")
        plot_two_distributions(avg_data1, avg_data2, data_labels, out_avg_plot, title="Average CC Value per Image")
        
        # All CC values from all images
        dist_data1 = [p[3] for p in aggregated_ccs[file1]]
        dist_data2 = [p[3] for p in aggregated_ccs[file2]]
        
        # Log the actual number of peaks in each distribution
        logging.info(f"Distribution statistics for {file1}: {len(dist_data1)} peaks from {img_count1} images")
        logging.info(f"Distribution statistics for {file2}: {len(dist_data2)} peaks from {img_count2} images")
        
        # Check expected peak counts (each image should contribute exactly npeaks)
        expected_peaks1 = img_count1 * (args.npeaks if args.npeaks else 0)
        expected_peaks2 = img_count2 * (args.npeaks if args.npeaks else 0)
        
        if expected_peaks1 > 0 and len(dist_data1) != expected_peaks1:
            logging.warning(f"Expected {expected_peaks1} peaks for {file1} (npeaks={args.npeaks} × {img_count1} images) but got {len(dist_data1)}")
        if expected_peaks2 > 0 and len(dist_data2) != expected_peaks2:
            logging.warning(f"Expected {expected_peaks2} peaks for {file2} (npeaks={args.npeaks} × {img_count2} images) but got {len(dist_data2)}")
            
        # Create the distribution plot of all peaks
        out_dist_plot = os.path.join(output_dir, "ccs_distribution_plot.png")
        plot_title = f"All CC Peak Values (top {args.npeaks} peaks from all images)"
        plot_two_distributions(dist_data1, dist_data2, data_labels, out_dist_plot, title=plot_title)
        
    else:
        # Single input file case
        file1 = os.path.splitext(os.path.basename(input_files[0]))[0]
        data_label = args.data_labels.split(',')[0]
        
        # Get the total number of images processed
        img_count1 = len(aggregated_avgs[file1])
        logging.info(f"Processed {img_count1} images from {file1}")
        
        # Average CC values per image
        avg_data = [a for (_, a) in aggregated_avgs[file1]]
        out_avg_plot = os.path.join(output_dir, "ccs_avgs_plot.png")
        plot_two_distributions(avg_data, None, [data_label], out_avg_plot, title="Average CC Value per Image")
        
        # All CC values from all images
        dist_data = [p[3] for p in aggregated_ccs[file1]]
        
        # Log the actual number of peaks in the distribution
        logging.info(f"Distribution statistics for {file1}: {len(dist_data)} peaks from {img_count1} images")
        
        # Check expected peak count
        expected_peaks = img_count1 * (args.npeaks if args.npeaks else 0)
        if expected_peaks > 0 and len(dist_data) != expected_peaks:
            logging.warning(f"Expected {expected_peaks} peaks for {file1} (npeaks={args.npeaks} × {img_count1} images) but got {len(dist_data)}")
            
        # Create the distribution plot of all peaks
        out_dist_plot = os.path.join(output_dir, "ccs_distribution_plot.png")
        plot_title = f"All CC Peak Values (top {args.npeaks} peaks from all images)"
        plot_two_distributions(dist_data, None, [data_label], out_dist_plot, title=plot_title)

    # Save CC maps
    if args.save_cc_maps > 0:
        cc_maps_by_file_img = defaultdict(dict)
        cc_maps_masked_by_file_img = defaultdict(dict)
        
        # Process cc maps and masked cc maps (if available)
        for key, value in cc_maps.items():
            if isinstance(value, tuple) and len(value) == 2:
                # We have both original and masked maps
                filename, image_idx, tmpl_idx = key
                cc_map_raw, cc_map_masked = value
                cc_maps_by_file_img[(filename, image_idx)][tmpl_idx] = cc_map_raw
                cc_maps_masked_by_file_img[(filename, image_idx)][tmpl_idx] = cc_map_masked
            else:
                # We just have the original map
                filename, image_idx, tmpl_idx = key
                cc_maps_by_file_img[(filename, image_idx)][tmpl_idx] = value
        
        # Group all maps by filename to save as stacks
        cc_maps_by_file = defaultdict(dict)
        cc_maps_masked_by_file = defaultdict(dict)
        
        # Process and organize maps into stacks by file
        for key, value in cc_maps.items():
            filename, image_idx, tmpl_idx = key
            if isinstance(value, tuple) and len(value) == 2:
                # We have both original and masked maps
                cc_map_raw, cc_map_masked = value
                cc_maps_by_file[filename][f"{image_idx}_{tmpl_idx}"] = cc_map_raw
                cc_maps_masked_by_file[filename][f"{image_idx}_{tmpl_idx}"] = cc_map_masked
            else:
                # We just have the original map
                cc_maps_by_file[filename][f"{image_idx}_{tmpl_idx}"] = value
        
        # Save original CC maps as one stack per file
        for filename, maps_dict in cc_maps_by_file.items():
            out_hdf = os.path.join(output_dir, f"{filename}_cc_maps.hdf")
            if EMAN2_AVAILABLE:
                save_cc_maps_stack_eman2(maps_dict, out_hdf)
            else:
                save_cc_maps_stack(maps_dict, out_hdf)
            logging.info(f"Saved {len(maps_dict)} CC maps for file {filename} to {out_hdf}")
                
        # Save masked CC maps for debugging (if available)
        for filename, maps_dict in cc_maps_masked_by_file.items():
            if maps_dict:  # Only if we have masked maps
                out_hdf = os.path.join(output_dir, f"{filename}_cc_maps_masked.hdf")
                if EMAN2_AVAILABLE:
                    save_cc_maps_stack_eman2(maps_dict, out_hdf)
                else:
                    save_cc_maps_stack(maps_dict, out_hdf)
                logging.info(f"Saved {len(maps_dict)} masked CC maps for file {filename} to {out_hdf}")

    # Build coordinate maps from final peaks for the first save_coords_map images only
    # This saves disk space while still processing all images for peak statistics
    coord_maps_by_file = defaultdict(list)
    for (filename, image_idx), final_peaks in final_peaks_by_img.items():
        # Only create and save coordinate maps for the first save_coords_map images per file
        if image_idx >= args.save_coords_maps:
            continue
            
        shape = shape_by_file_img[(filename, image_idx)]
        coords_map = np.zeros(shape, dtype=np.uint8)
        for (x, y, z, val) in final_peaks:
            place_sphere(coords_map, (x, y, z), args.coords_map_r)
        coord_maps_by_file[filename].append(coords_map)
        logging.info(f"Created coordinate map for file {filename}, image {image_idx} with {len(final_peaks)} peaks")

    # Save coordinate maps
    for filename, maps_list in coord_maps_by_file.items():
        # Ensure we only save up to save_coords_map maps per file
        maps_to_save = maps_list[:args.save_coords_maps]
        out_map = os.path.join(output_dir, f"{filename}_coords_map.hdf")
        save_coords_map_as_hdf(maps_to_save, out_map)
        logging.info(f"Saved {len(maps_to_save)} coordinate map(s) for file {filename} to {out_map}")

    # Optionally, save CSV
    if args.save_csv:
        csv_out = os.path.join(output_dir, "aggregated_cc_peaks.csv")
        with open(csv_out, 'w', newline='') as csvfile:
            fieldnames = ["Input_File", "Image_Index", "X", "Y", "Z", "CC_Value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for (filename, image_idx), final_peaks in final_peaks_by_img.items():
                for (x, y, z, val) in final_peaks:
                    writer.writerow({
                        "Input_File": filename,
                        "Image_Index": image_idx,
                        "X": x,
                        "Y": y,
                        "Z": z,
                        "CC_Value": f"{val:.6f}"
                    })
        logging.info(f"Saved aggregated CSV summary of all extracted peaks to {csv_out}")

    overall_end = time.time()
    logging.info(f"Total processing time: {overall_end - overall_start:.2f} seconds")
    logging.info(f"Final memory usage: {process_mem.memory_info().rss / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Fatal error: " + str(e))
        logging.error(traceback.format_exc())
        sys.exit(1)
        '''