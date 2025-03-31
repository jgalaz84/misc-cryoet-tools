#!/usr/bin/env python

import argparse
import os
import sys
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

def load_image(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.hdf', '.h5']:
            with h5py.File(file_path, 'r') as file:
                dataset_paths = [f"MDF/images/{i}/image" for i in range(len(file["MDF/images"]))]
                images = [file[ds][:] for ds in dataset_paths]
            return images
        elif ext == '.mrc':
            with mrcfile.open(file_path, permissive=True) as mrc:
                data = mrc.data
                if data.ndim == 3:
                    return [data]
                elif data.ndim >= 4:
                    return [d for d in data]
                else:
                    return [data]
        else:
            raise ValueError(f"Unsupported file format: {ext}")
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

#########################
# Worker function
#########################

def process_image_template(args_tuple):
    """
    1) Compute CC map,
    2) Use peak finding method as specified by the user,
    3) Return final picks + optional CC map if store_ccmap==True.
    
    This function supports three peak-finding methods:
    - "original": Uses the original reliable NMS algorithm
    - "eman2": Uses EMAN2's masking approach for NMS
    - "fallback": Uses a pure-Python approach, no EMAN2 required
    """
    try:
        (filename, image_idx, template_idx, image, template,
         n_peaks_local, cc_thresh, diameter, store_ccmap, method, template_path) = args_tuple

        # 1) Compute normalized cross-correlation map
        cc_map = compute_ncc_map_3d_single(image, template)

        # 2) Find peaks using appropriate method
        if method == "eman2" and EMAN2_AVAILABLE:
            # Use EMAN2 for peak finding - note that this uses the "outer_radius" parameter 
            # which is set to radius (half of diameter)
            radius = diameter / 2.0
            logging.debug(f"Using EMAN2 method with mask.sharp outer_radius={radius} (diameter={diameter})")
            local_peaks, cc_map_masked = iterative_nms_eman2(cc_map, n_peaks_local, diameter, cc_thresh, verbose=_VERBOSE)
            
            # Save masked CC map for debugging EMAN2 method
            if store_ccmap and template_idx == 0:
                # Return both the original cc_map and masked map for debugging
                return (filename, image_idx, template_idx, local_peaks, cc_map, cc_map_masked)
                
        elif method == "fallback" or (method == "eman2" and not EMAN2_AVAILABLE):
            # Use fallback method when EMAN2 is requested but not available,
            # or when fallback is explicitly requested
            logging.debug(f"Using fallback method (no EMAN2) with diameter={diameter}")
            local_peaks, cc_map_masked = fallback_iterative_nms(cc_map, n_peaks_local, diameter, cc_thresh, verbose=_VERBOSE)
        else:
            # Default to original NMS algorithm - most reliable for n_peaks
            logging.debug(f"Using original method with diameter={diameter}")
            if cc_thresh > 0:
                thresh = cc_map.mean() + cc_thresh * cc_map.std()
                cc_map_thresholded = np.where(cc_map >= thresh, cc_map, -np.inf)
                local_peaks = non_maximum_suppression(cc_map_thresholded, n_peaks_local, diameter)
            else:
                local_peaks = non_maximum_suppression(cc_map, n_peaks_local, diameter)
            cc_map_masked = None  # Not needed for original method

        # Log how many peaks were actually found
        logging.debug(f"Found {len(local_peaks)} peaks for file {filename}, image {image_idx}, template {template_idx}")

        # Dump peak coordinates for debugging (just for the first template)
        if template_idx == 0:
            logging.info(f"PEAK COORDINATES for {filename}, image {image_idx}, template {template_idx}:")
            for i, (x, y, z, val) in enumerate(local_peaks[:min(10, len(local_peaks))]):  # Show first 10 peaks
                logging.info(f"  Peak #{i+1}: (x={x}, y={y}, z={z}), val={val:.3f}")
            if len(local_peaks) > 10:
                logging.info(f"  ... and {len(local_peaks)-10} more peaks")

        # local_peaks are in format (x, y, z, val)
        if store_ccmap:
            return (filename, image_idx, template_idx, local_peaks, cc_map)
        else:
            return (filename, image_idx, template_idx, local_peaks)

    except Exception as e:
        logging.error(f"Error processing file {filename} image {image_idx} template {template_idx}: {e}")
        logging.error(traceback.format_exc())
        return None

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
    parser = argparse.ArgumentParser(description="Compute 3D CC maps and extract peaks using non-maximum suppression.")

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
    parser.add_argument('--save_ccmaps', type=int, default=1,
                        help="How many CC maps to save per input file. Default=1.")
    parser.add_argument('--save_coords_map', type=int, default=1,
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
    if args.save_ccmaps < 1:
        logging.error("--save_ccmaps must be >= 1.")
        sys.exit(1)
        
    if args.save_coords_map < 1:
        logging.error("--save_coords_map must be >= 1.")
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
    if args.save_ccmaps > n_templates:
        logging.info(f"--save_ccmaps={args.save_ccmaps} but only {n_templates} template(s) present. Reducing to {n_templates}.")
        args.save_ccmaps = n_templates

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
    if args.save_coords_map > min_images:
        logging.info(f"--save_coords_map={args.save_coords_map} but minimum image count is {min_images}. Reducing to {min_images}.")
        args.save_coords_map = min_images

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
        
        # Only process up to save_coords_map images for each file
        for img_idx, image_3d in enumerate(image_list[:args.save_coords_map]):
            shape_by_file_img[(filename, img_idx)] = image_3d.shape

            # Calculate local_npeaks (how many peaks to find per template)
            if args.npeaks is None:
                diam_int = int(args.diameter)
                n_possible = (image_3d.shape[0] // diam_int) * (image_3d.shape[1] // diam_int) * (image_3d.shape[2] // diam_int)
                local_npeaks = n_possible
            else:
                local_npeaks = args.npeaks

            # For each template
            for tmpl_idx, template_data in enumerate(templates):
                store_map = (tmpl_idx < args.save_ccmaps)  # only store CC map for first N templates
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

    # Plot distributions
    if len(input_files) == 2:
        data_labels = args.data_labels.split(',')
        file1 = os.path.splitext(os.path.basename(input_files[0]))[0]
        file2 = os.path.splitext(os.path.basename(input_files[1]))[0]
        
        # Average CC values per image
        avg_data1 = [a for (_, a) in aggregated_avgs[file1]]
        avg_data2 = [a for (_, a) in aggregated_avgs[file2]]
        out_avg_plot = os.path.join(output_dir, "ccs_avgs_plot.png")
        plot_two_distributions(avg_data1, avg_data2, data_labels, out_avg_plot, title="Average CC Value per Image")
        
        # All CC values - no sampling/subsampling to ensure we use all peaks
        dist_data1 = [p[3] for p in aggregated_ccs[file1]]
        dist_data2 = [p[3] for p in aggregated_ccs[file2]]
        
        # Log the actual number of peaks in each distribution
        logging.info(f"Distribution statistics for {file1}: {len(dist_data1)} peaks from {len(avg_data1)} images")
        logging.info(f"Distribution statistics for {file2}: {len(dist_data2)} peaks from {len(avg_data2)} images")
        
        # Check expected peak counts
        expected_peaks1 = len(avg_data1) * (args.npeaks if args.npeaks else 0)
        expected_peaks2 = len(avg_data2) * (args.npeaks if args.npeaks else 0)
        
        if expected_peaks1 > 0 and len(dist_data1) != expected_peaks1:
            logging.warning(f"Expected {expected_peaks1} peaks for {file1} (npeaks={args.npeaks} × {len(avg_data1)} images) but got {len(dist_data1)}")
        if expected_peaks2 > 0 and len(dist_data2) != expected_peaks2:
            logging.warning(f"Expected {expected_peaks2} peaks for {file2} (npeaks={args.npeaks} × {len(avg_data2)} images) but got {len(dist_data2)}")
            
        out_dist_plot = os.path.join(output_dir, "ccs_distribution_plot.png")
        plot_two_distributions(dist_data1, dist_data2, data_labels, out_dist_plot, title="All CC Peak Values")
    else:
        file1 = os.path.splitext(os.path.basename(input_files[0]))[0]
        data_label = args.data_labels.split(',')[0]
        
        # Average CC values per image
        avg_data = [a for (_, a) in aggregated_avgs[file1]]
        out_avg_plot = os.path.join(output_dir, "ccs_avgs_plot.png")
        plot_two_distributions(avg_data, None, [data_label], out_avg_plot, title="Average CC Value per Image")
        
        # All CC values
        dist_data = [p[3] for p in aggregated_ccs[file1]]
        
        # Log the actual number of peaks in the distribution
        logging.info(f"Distribution statistics for {file1}: {len(dist_data)} peaks from {len(avg_data)} images")
        
        # Check expected peak count
        expected_peaks = len(avg_data) * (args.npeaks if args.npeaks else 0)
        if expected_peaks > 0 and len(dist_data) != expected_peaks:
            logging.warning(f"Expected {expected_peaks} peaks for {file1} (npeaks={args.npeaks} × {len(avg_data)} images) but got {len(dist_data)}")
            
        out_dist_plot = os.path.join(output_dir, "ccs_distribution_plot.png")
        plot_two_distributions(dist_data, None, [data_label], out_dist_plot, title="All CC Peak Values")

    # Save CC maps
    if args.save_ccmaps > 0:
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

    # Always build coordinate maps from final peaks (using coords_map_r parameter)
    coord_maps_by_file = defaultdict(list)
    for (filename, image_idx), final_peaks in final_peaks_by_img.items():
        # Only include up to save_coords_map images per file
        if image_idx >= args.save_coords_map:
            continue
            
        shape = shape_by_file_img[(filename, image_idx)]
        coords_map = np.zeros(shape, dtype=np.uint8)
        for (x, y, z, val) in final_peaks:
            place_sphere(coords_map, (x, y, z), args.coords_map_r)  # Use the new parameter
        coord_maps_by_file[filename].append(coords_map)
        logging.info(f"Created coordinate map for file {filename}, image {image_idx} with {len(final_peaks)} peaks")

    # Save coordinate maps
    for filename, maps_list in coord_maps_by_file.items():
        # Ensure we only save up to save_coords_map maps per file
        maps_to_save = maps_list[:args.save_coords_map]
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