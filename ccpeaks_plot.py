#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2025
# Last modification: Mar/21/2025
#
# FEATURES:
#  1) Computes 3D cross-correlation (CC) maps between tomograms and one or more templates.
#  2) Uses a greedy non-maximum suppression (NMS) algorithm:
#       - Finds the highest voxel in the CC map,
#       - Ignores all voxels within --diameter of that maximum,
#       - Then finds the next highest surviving voxel,
#       - Repeats until --npeaks are found (if --npeaks > 0; if 0, keep all).
#  3) Does NOT apply any threshold by default (i.e. --ccc_thresh defaults to 0).
#  4) Saves per-template results (in "correlation_files") and combined final peaks (in "raw_coordinates").
#  5) Optionally saves the full CC maps (now referred to as "cc_maps") for each (image,template) pair.
#  6) Extensive inline comments, verbose logging, and progress bars via tqdm.
#
# USAGE (example):
#   python ccpeaks_plot.py --input tomo1.mrc,tomo2.hdf --template templ.hdf \
#      --output_dir my_output --npeaks 34 --diameter 38 --save_coords_map_r 4 --ccc_thresh 0 --verbose 2
#
# NOTE: Requires tqdm (pip install tqdm)
#

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
from tqdm import tqdm  # For progress bar

#########################
# Utility functions
#########################

def create_output_directory(base_dir):
    """
    Create a new output directory by appending _00, _01, etc.
    Also creates subfolders: correlation_files, coordinate_plots, and raw_coordinates.
    Returns the created directory path.
    """
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
    """
    Setup logging to both a file (in output_dir) and console.
    Verbosity:
      0 -> ERROR, 1 -> WARNING, 2 -> INFO, 3 -> DEBUG.
    """
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
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Command: " + " ".join(sys.argv))
    return log_file

def parse_mrc_apix_and_mean(file_path):
    """
    Attempt to parse pixel size (apix) and compute mean intensity for an MRC file.
    Returns (apix, mean_intensity) or (None, None) if unavailable.
    """
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
    """
    Attempt to parse pixel size (apix) and compute mean intensity for an HDF file.
    Returns (apix, mean_intensity) or (None, None) if not found.
    """
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
    """
    Load a 3D image or stack from an MRC or HDF file.
    Returns the image array, or None on error.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.hdf', '.h5']:
            with h5py.File(file_path, 'r') as file:
                dataset_paths = [f"MDF/images/{i}/image" for i in range(len(file["MDF/images"]))]
                image_stack = np.array([file[ds][:] for ds in dataset_paths])
        elif ext == '.mrc':
            with mrcfile.open(file_path, permissive=True) as mrc:
                image_stack = mrc.data
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None
    return image_stack

def check_apix_and_intensity(tomo_path, template_path):
    """
    Compare approximate pixel size (apix) and mean intensity for a tomogram vs. a template.
    Logs a warning if they differ by more than a factor of 2.
    """
    tomo_ext = os.path.splitext(tomo_path)[1].lower()
    template_ext = os.path.splitext(template_path)[1].lower()
    if tomo_ext == '.mrc':
        tomo_apix, tomo_mean = parse_mrc_apix_and_mean(tomo_path)
    elif tomo_ext in ['.hdf', '.h5']:
        tomo_apix, tomo_mean = parse_hdf_apix_and_mean(tomo_path)
    else:
        tomo_apix, tomo_mean = (None, None)
    if template_ext == '.mrc':
        tpl_apix, tpl_mean = parse_mrc_apix_and_mean(template_path)
    elif template_ext in ['.hdf', '.h5']:
        tpl_apix, tpl_mean = parse_hdf_apix_and_mean(template_path)
    else:
        tpl_apix, tpl_mean = (None, None)
    if tomo_apix and tpl_apix:
        ratio = tpl_apix / tomo_apix if tomo_apix != 0 else None
        if ratio and (ratio < 0.5 or ratio > 2.0):
            logging.warning(f"Template apix {tpl_apix:.2f} vs. tomogram apix {tomo_apix:.2f} differs by >2x.")
    if tomo_mean and tpl_mean:
        ratio = tpl_mean / tomo_mean if tomo_mean != 0 else None
        if ratio and (ratio < 0.5 or ratio > 2.0):
            logging.warning(f"Template mean intensity {tpl_mean:.2f} vs. tomogram mean intensity {tomo_mean:.2f} differs by >2x.")

def adjust_template(template, target_shape):
    """
    Adjust the template to match the target shape by centering and clipping or padding.
    Returns a new array of shape 'target_shape'.
    """
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
    """
    Compute the 3D normalized cross-correlation (CC) map between image and template.
    If shapes differ, adjusts the template.
    Returns the CC map.
    """
    if image.shape != template.shape:
        template = adjust_template(template, image.shape)
    cc_map = correlate(image, template, mode='same', method='fft')
    cc_map = (cc_map - np.mean(cc_map)) / np.std(cc_map)
    return cc_map

def non_maximum_suppression(cc_map, n_peaks, diameter):
    """
    Perform greedy non-maximum suppression on the CC map:
      1. Flatten and sort all voxels in descending order.
      2. Iterate over sorted candidates; accept a candidate if it is not within
         'diameter' voxels of any already accepted candidate.
      3. Stop when n_peaks candidates have been accepted (if n_peaks > 0).
         If n_peaks==0, return all candidates.
    Returns a list of (x, y, z, value).
    """
    flat = cc_map.ravel()
    sorted_indices = np.argsort(flat)[::-1]
    coords = np.column_stack(np.unravel_index(sorted_indices, cc_map.shape))
    accepted = []
    for idx, coord in enumerate(coords):
        candidate_val = flat[sorted_indices[idx]]
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

#########################
# Output saving functions
#########################

def save_peak_data(output_dir, filename, template_str, image_idx, peak_coords, peak_values, total_templates):
    """
    Save the final peaks for one (image, template) pair into a text file in "correlation_files".
    Columns: Template, X, Y, Z, CC_Value.
    Note: All instances of "ccc" have been renamed to "cc".
    """
    image_str = str(image_idx).zfill(3)
    out_file = os.path.join(output_dir, 'correlation_files', f'cc_{filename}_T{template_str}_I{image_str}.txt')
    header = f"{'Template':<10} {'X':>5} {'Y':>5} {'Z':>5} {'CC_Value':>10}"
    with open(out_file, 'w') as f:
        f.write(header + "\n")
        for i in range(len(peak_values)):
            x, y, z = map(int, peak_coords[i])
            val = float(peak_values[i])
            f.write(f"{template_str:<10} {x:5d} {y:5d} {z:5d} {val:10.6f}\n")
    logging.info(f"Saved CC peaks to {out_file}")

def plot_3d_peak_coordinates(output_dir, filename, template_str, image_idx, peak_coords):
    """
    Create a 3D scatter plot of the peaks for one (image, template) pair and save as PNG in "coordinate_plots".
    """
    if peak_coords.size == 0:
        return
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peak_coords[:,0], peak_coords[:,1], peak_coords[:,2], c='red', marker='o')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Top CC Peak Locations")
    image_str = str(image_idx).zfill(3)
    out_png = os.path.join(output_dir, 'coordinate_plots', f'cc_{filename}_T{template_str}_I{image_str}.png')
    # Note: compress_level is no longer supported by savefig in newer matplotlib.
    plt.savefig(out_png, dpi=150)
    plt.close()
    logging.info(f"Saved 3D peak plot to {out_png}")

def place_sphere(volume, center, radius):
    """
    Mark a sphere (of ones) in 'volume' at 'center' with integer radius 'radius'.
    Used for generating the coords_map.
    """
    x0, y0, z0 = center
    nx, ny, nz = volume.shape
    for x in range(x0 - radius, x0 + radius + 1):
        if x < 0 or x >= nx:
            continue
        for y in range(y0 - radius, y0 + radius + 1):
            if y < 0 or y >= ny:
                continue
            for z in range(z0 - radius, z0 + radius + 1):
                if z < 0 or z >= nz:
                    continue
                if (x - x0)**2 + (y - y0)**2 + (z - z0)**2 <= radius**2:
                    volume[x, y, z] = 1

def save_coords_map_as_hdf(coords_map_stack, out_path):
    """
    Save a list of 3D volumes (coords_map_stack) to an HDF file in EMAN2 style:
      /MDF/images/0/image, /MDF/images/1/image, etc.
    """
    with h5py.File(out_path, 'w') as f:
        for i in range(len(coords_map_stack)):
            ds_name = f"MDF/images/{i}/image"
            f.create_dataset(ds_name, data=coords_map_stack[i], dtype=coords_map_stack[i].dtype)
    logging.info(f"Saved coords_map to {out_path}")

def save_cc_maps_stack(cc_maps_dict, out_path):
    """
    Save full CC maps from all templates for a given (filename, image_idx) into an HDF stack.
    The maps are saved in order (template 0, 1, 2, ...).
    """
    sorted_keys = sorted(cc_maps_dict.keys())
    stack = np.stack([cc_maps_dict[k] for k in sorted_keys], axis=0)
    with h5py.File(out_path, 'w') as f:
        for i in range(stack.shape[0]):
            ds_name = f"MDF/images/{i}/image"
            f.create_dataset(ds_name, data=stack[i], dtype=stack.dtype)
    logging.info(f"Saved CC maps stack to {out_path}")

#########################
# Worker function
#########################

def process_image_template(args_tuple):
    """
    Worker function that:
      1. Computes the CC map for one (image, template) pair.
      2. (Optionally) applies a threshold if ccc_thresh_sigma > 0.
      3. Uses non-maximum suppression to select peaks: it finds the global maximum,
         then disregards all voxels within --diameter, then repeats.
      4. Returns the candidate peaks (each as (x, y, z, value)).
         If save_cc_maps is True, also returns the full CC map.
    """
    try:
        (filename, image_idx, template_idx, image, template,
         n_peaks, ccc_thresh_sigma, diameter, save_cc_maps) = args_tuple
        cc_map = compute_ncc_map_3d_single(image, template)
        # If a threshold is specified, zero out values below threshold (so they won't be picked)
        if ccc_thresh_sigma > 0:
            thresh = cc_map.mean() + ccc_thresh_sigma * cc_map.std()
            cc_map = np.where(cc_map >= thresh, cc_map, -np.inf)
        # Apply greedy non-maximum suppression to extract exactly n_peaks (if n_peaks>0)
        candidates = non_maximum_suppression(cc_map, n_peaks, diameter)
        if save_cc_maps:
            return (filename, image_idx, template_idx, candidates, cc_map)
        else:
            return (filename, image_idx, template_idx, candidates)
    except Exception as e:
        logging.error("Error processing file {} image {} template {}: {}".format(
            filename, image_idx, template_idx, str(e)))
        logging.error(traceback.format_exc())
        return None

#########################
# Main function
#########################

def main():
    """
    Main routine:
      - Parse command-line arguments.
      - Load templates and tomograms.
      - Check pixel size and intensity.
      - Compute auto diameter if needed.
      - Build tasks for each (file, image, template) pair.
      - Run tasks in parallel with a progress bar.
      - For each (file, image, template), apply non-maximum suppression and save per-template results.
      - Combine results across templates for each image and save raw coordinates.
      - Optionally, build and save coords_map volumes and full CC maps stacks.
    """
    parser = argparse.ArgumentParser(
        description="Compute 3D CC maps and extract peaks using non-maximum suppression."
    )
    parser.add_argument('--input', required=True,
                        help="Comma-separated input file paths (tomograms).")
    parser.add_argument('--template', required=True,
                        help="Template image file path (can be an HDF stack).")
    parser.add_argument('--npeaks', type=int, default=34,
                        help="Number of peaks to keep after NMS. If 0, keep all.")
    parser.add_argument('--diameter', type=float, default=38,
                        help="Minimum allowed distance between peaks (in voxels).")
    parser.add_argument('--threads', type=int, default=1,
                        help="Number of processes to use in parallel.")
    parser.add_argument('--output_dir', default="cc_analysis",
                        help="Base name for output directory.")
    parser.add_argument('--save_coords_map_r', type=int, default=0,
                        help="If >0, produce an HDF volume marking each final peak with a sphere of radius r.")
    # Add an alias so that --ccc_thresh can be used; internal name remains ccc_thresh_sigma.
    parser.add_argument('--cc_thresh', dest='ccc_thresh_sigma', type=float, default=0.0,
                    help="Threshold for picking peaks: mean + sigma*std. If <=0, no threshold is applied. Default=0.")
    parser.add_argument('--save_cc_maps', action='store_true', default=False,
                        help="If set, save full CC maps for each (image, template) pair into a stack (cc_maps.hdf).")
    parser.add_argument('--verbose', type=int, default=2,
                        help="Verbosity level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG.")
    
    args = parser.parse_args()
    
    # Create output directory and setup logging.
    output_dir = create_output_directory(args.output_dir)
    setup_logging(output_dir, args.verbose)
    
    start_time = time.time()
    logging.info("Starting processing...")
    
    # Load templates.
    templates = load_image(args.template)
    if templates is None:
        logging.error("Could not load template file. Exiting.")
        sys.exit(1)
    templates = templates.squeeze()
    if templates.ndim == 3:
        templates = np.expand_dims(templates, axis=0)
    n_templates = templates.shape[0]
    n_digits = len(str(n_templates))
    logging.info(f"Loaded {n_templates} template(s) from {args.template}")
    
    # Parse input tomogram files.
    input_files = args.input.split(',')
    
    # Get shape of first tomogram (for auto-diameter if needed).
    first_tomo_file = input_files[0]
    first_image_stack = load_image(first_tomo_file)
    if first_image_stack is None:
        logging.error(f"Could not load first tomogram {first_tomo_file}. Exiting.")
        sys.exit(1)
    if first_image_stack.ndim == 3:
        first_image_stack = np.expand_dims(first_image_stack, axis=0)
    first_tomo_shape = first_image_stack[0].shape
    
    # (If diameter <= 0, auto-derive; here we expect a positive diameter from the user.)
    if args.diameter <= 0:
        auto_diameter = compute_auto_diameter(templates[0], first_tomo_shape)
        args.diameter = auto_diameter
        logging.info(f"No diameter provided; using auto-derived diameter: {auto_diameter}")
    
    # Build tasks for each (file, image, template) pair.
    tasks = []
    shape_by_file_img = {}  # For later use in coords_map.
    for file_path in input_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        check_apix_and_intensity(file_path, args.template)
        image_stack = load_image(file_path)
        if image_stack is None:
            logging.error(f"File {file_path} could not be loaded. Skipping.")
            continue
        if image_stack.ndim == 3:
            image_stack = np.expand_dims(image_stack, axis=0)
        n_images = image_stack.shape[0]
        for img_idx in range(n_images):
            shape_by_file_img[(filename, img_idx)] = image_stack[img_idx].shape
            for tmpl_idx in range(n_templates):
                tasks.append((
                    filename, img_idx, tmpl_idx,
                    image_stack[img_idx], templates[tmpl_idx],
                    args.npeaks, args.ccc_thresh_sigma, args.diameter, args.save_cc_maps
                ))
    
    # Run tasks in parallel with tqdm progress bar.
    partial_results = []
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_image_template, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
            res = fut.result()
            if res is not None:
                partial_results.append(res)
    
    # Organize worker results by (filename, image_idx, template_idx).
    partial_peaks = defaultdict(list)
    cc_maps = {}  # Only used if save_cc_maps is True.
    for res in partial_results:
        if args.save_cc_maps:
            filename, image_idx, template_idx, peaks, cc_map = res
            cc_maps[(filename, image_idx, template_idx)] = cc_map
        else:
            filename, image_idx, template_idx, peaks = res
        partial_peaks[(filename, image_idx, template_idx)] = peaks
        logging.debug(f"File {filename}, image {image_idx}, template {template_idx} returned {len(peaks)} peaks after NMS.")
    
    # 1) For each (file, image, template), save per-template results.
    for key, peaks in partial_peaks.items():
        filename, image_idx, template_idx = key
        filtered = peaks  # Our NMS algorithm already returns mutually exclusive peaks.
        template_str = str(template_idx).zfill(n_digits)
        peak_coords = np.array([[p[0], p[1], p[2]] for p in filtered], dtype=int)
        peak_values = np.array([p[3] for p in filtered], dtype=float)
        save_peak_data(output_dir, filename, template_str, image_idx, peak_coords, peak_values, n_templates)
        if peak_coords.size > 0:
            plot_3d_peak_coordinates(output_dir, filename, template_str, image_idx, peak_coords)
        partial_peaks[key] = filtered  # Update (not strictly needed)
    
    # 2) Combine peaks across templates for each (file, image) and save raw coordinates.
    combined_peaks_by_file_img = defaultdict(list)
    for (filename, image_idx, template_idx), peaks in partial_peaks.items():
        combined_peaks_by_file_img[(filename, image_idx)].extend(peaks)
    for (filename, image_idx), peaks in combined_peaks_by_file_img.items():
        final_peaks = non_maximum_suppression_from_list(peaks, args.npeaks, args.diameter)
        raw_coords_dir = os.path.join(output_dir, "raw_coordinates")
        img_str = str(image_idx).zfill(2)
        out_txt = os.path.join(raw_coords_dir, f"coords_{filename}_{img_str}.txt")
        with open(out_txt, 'w') as f:
            for (x, y, z, val) in final_peaks:
                f.write(f"{int(x):6d} {int(y):6d} {int(z):6d}\n")
        logging.info(f"Saved raw coords to {out_txt}")
        combined_peaks_by_file_img[(filename, image_idx)] = final_peaks
    
    # 3) If requested, build and save coords_map volumes for each file.
    if args.save_coords_map_r > 0:
        file_to_num_images = defaultdict(int)
        for file_path in input_files:
            fname = os.path.splitext(os.path.basename(file_path))[0]
            image_stack = load_image(file_path)
            if image_stack is None:
                logging.error(f"Error re-loading file {file_path} for coords_map; skipping.")
                continue
            if image_stack.ndim == 3:
                image_stack = np.expand_dims(image_stack, axis=0)
            file_to_num_images[fname] = image_stack.shape[0]
        for file_path in input_files:
            fname = os.path.splitext(os.path.basename(file_path))[0]
            if fname not in file_to_num_images:
                continue
            n_images = file_to_num_images[fname]
            coords_map_stack = []
            for i in range(n_images):
                shape_3d = shape_by_file_img[(fname, i)]
                coords_map_stack.append(np.zeros(shape_3d, dtype=np.uint8))
            for i in range(n_images):
                final_peaks = combined_peaks_by_file_img[(fname, i)]
                for (x, y, z, val) in final_peaks:
                    place_sphere(coords_map_stack[i], (int(x), int(y), int(z)), args.save_coords_map_r)
            out_hdf = os.path.join(output_dir, f"{fname}_coords_map.hdf")
            save_coords_map_as_hdf(coords_map_stack, out_hdf)
    # 4) If requested, save full CC maps for each (file, image) into a stack.
    if args.save_cc_maps:
        cc_maps_by_file_img = defaultdict(dict)
        for (filename, image_idx, template_idx), cc_map in cc_maps.items():
            cc_maps_by_file_img[(filename, image_idx)][template_idx] = cc_map
        for (filename, image_idx), maps_dict in cc_maps_by_file_img.items():
            out_hdf = os.path.join(output_dir, f"{filename}_cc_maps.hdf")
            save_cc_maps_stack(maps_dict, out_hdf)
    
    elapsed = time.time() - start_time
    logging.info(f"Processing finished successfully in {elapsed:.2f} seconds.")

def non_maximum_suppression_from_list(candidate_list, n_peaks, diameter):
    """
    Given a list of candidate peaks (each as (x,y,z,value)), perform greedy non-maximum suppression.
    This function is used to combine candidates from multiple templates.
    Returns a list of peaks (x,y,z,value) that are at least 'diameter' apart, up to n_peaks.
    If n_peaks==0, return all.
    """
    # Sort candidates by value descending
    candidate_list.sort(key=lambda x: x[3], reverse=True)
    accepted = []
    for cand in candidate_list:
        x, y, z, val = cand
        too_close = False
        for (ax, ay, az, aval) in accepted:
            if np.linalg.norm(np.array([x, y, z]) - np.array([ax, ay, az])) < diameter:
                too_close = True
                break
        if not too_close:
            accepted.append(cand)
        if n_peaks > 0 and len(accepted) >= n_peaks:
            break
    return accepted

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Fatal error: " + str(e))
        logging.error(traceback.format_exc())
        sys.exit(1)





'''
#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2025; last modification: Mar/21/2025
#
# Modifications:
#  1. Added logging and a log file that records the full command for reproducibility.
#  2. Renamed 'coordinate_files' to 'coordinate_plots' and saved PNGs with compression.
#  3. Added support for multiple templates as an HDF stack.
#  4. Added --diameter filtering to ensure peaks are not too close.
#  5. Parallelized the NCC computation over image-template pairs using --threads.
#  6. Fixed dtype mismatch in saving peaks by writing rows as text manually.
#  7. **NEW**: If --diameter <= 0, automatically derive a diameter from the template’s
#     bounding box of meaningful density, clamped to the shortest dimension of the first
#     tomogram and the template.

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

def create_output_directory(base_dir):
    """Create and return a numbered output directory with needed subfolders."""
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "correlation_files"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "coordinate_plots"), exist_ok=True)
    return output_dir

def setup_logging(output_dir):
    """Set up logging to a file and console."""
    log_file = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Command: " + " ".join(sys.argv))
    return log_file

def load_image(file_path):
    """Loads a 3D image or a stack of 3D images from an HDF5 or MRC file."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            # Assumes structure MDF/images/<i>/image
            dataset_paths = [f"MDF/images/{i}/image" for i in range(len(file["MDF/images"]))]
            image_stack = np.array([file[ds][:] for ds in dataset_paths])
    elif ext == '.mrc':
        with mrcfile.open(file_path, permissive=True) as mrc:
            image_stack = mrc.data
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return image_stack

def adjust_template(template, target_shape):
    """
    Adjusts the template to match the target shape (tomogram dimensions)
    by centering and clipping or padding as necessary.
    """
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
    """
    Computes the 3D normalized cross-correlation map for one image and one template.
    If the template shape does not match the image shape, adjust it accordingly.
    """
    if image.shape != template.shape:
        template = adjust_template(template, image.shape)
    ncc = correlate(image, template, mode='same', method='fft')
    ncc = (ncc - np.mean(ncc)) / np.std(ncc)
    return ncc

def extract_ncc_peaks_with_diameter(ncc_map, npeaks, diameter):
    """
    Extracts up to npeaks from the NCC map ensuring that no two accepted peaks
    are closer than 'diameter'. Peaks are taken in descending order of NCC value.
    """
    sorted_indices = np.argsort(ncc_map.ravel())[::-1]
    accepted_coords = []
    accepted_values = []
    for idx in sorted_indices:
        coord = np.array(np.unravel_index(idx, ncc_map.shape))
        value = ncc_map.ravel()[idx]
        if diameter > 0 and accepted_coords:
            dists = np.linalg.norm(np.array(accepted_coords) - coord, axis=1)
            if np.any(dists < diameter):
                continue
        accepted_coords.append(coord)
        accepted_values.append(value)
        if len(accepted_coords) >= npeaks:
            break
    return np.array(accepted_values), np.array(accepted_coords)

def compute_auto_diameter(template, tomo_shape):
    """
    1) Low-pass filter (Gaussian blur, sigma=2).
    2) Threshold at mean + 2*std to form a mask of “meaningful” density.
    3) Find bounding box of that mask, define “longest span” = max dimension.
    4) Clamp so it cannot exceed the shortest dimension of the tomogram
       or the shortest dimension of the template.
    """
    blurred = gaussian_filter(template, sigma=2.0)
    mean_val = blurred.mean()
    std_val = blurred.std()
    threshold = mean_val + 2.0 * std_val

    mask = (blurred >= threshold)
    if not np.any(mask):
        logging.warning("Auto diameter: no voxels above threshold, returning diameter=0.")
        return 0.0

    coords = np.argwhere(mask)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    # bounding box size
    dx = x_max - x_min + 1
    dy = y_max - y_min + 1
    dz = z_max - z_min + 1
    diameter = max(dx, dy, dz)

    # clamp to the smaller of: min dimension of tomo or min dimension of template
    diameter = min(diameter, min(tomo_shape))
    diameter = min(diameter, min(template.shape))

    return diameter

def save_peak_data(output_dir, filename, template_str, image_idx, peak_coords, peak_values, total_templates):
    """
    Saves the peak coordinates and NCC values to a formatted text file.
    Columns: Template, X, Y, Z, NCC_Value
    """
    image_str = str(image_idx).zfill(3)
    out_file = os.path.join(output_dir, 'correlation_files', f'ccc_{filename}_T{template_str}_I{image_str}.txt')
    
    # We'll define column widths so things line up under the header
    header = f"{'Template':<10} {'X':>5} {'Y':>5} {'Z':>5} {'NCC_Value':>10}"
    
    with open(out_file, 'w') as f:
        f.write(header + "\n")
        for i in range(len(peak_values)):
            x, y, z = map(int, peak_coords[i])
            val = float(peak_values[i])
            f.write(f"{template_str:<10} {x:5d} {y:5d} {z:5d} {val:10.6f}\n")
    
    logging.info(f"Saved NCC peaks to {out_file}")

def plot_3d_peak_coordinates(output_dir, filename, template_str, image_idx, peak_coords):
    """Plots the NCC peak coordinates in 3D and saves the plot as a compressed PNG."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2], c='red', marker='o')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Top NCC Peak Locations")
    image_str = str(image_idx).zfill(3)
    out_png = os.path.join(output_dir, 'coordinate_plots', f'ccc_{filename}_T{template_str}_I{image_str}.png')
    plt.savefig(out_png, dpi=150, compress_level=9)
    plt.close()
    logging.info(f"Saved 3D peak plot to {out_png}")

def is_normal(data, alpha=0.05):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    _, p_value = shapiro(data)
    return p_value > alpha

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
    """
    Generates a violin plot comparing NCC peak distributions with relevant statistics.
    Displays total number of points (N) for each distribution above its violin.
    """
    p_value, effect_size, method = calculate_effect_size(data1, data2)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the two distributions as a violin plot
    parts = ax.violinplot([data1, data2], showmeans=True, showmedians=True)

    # Customize violin colors
    colors = ['blue', 'orange']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
    
    # Style mean/median lines
    if 'cmeans' in parts:
        parts['cmeans'].set_linestyle('--')
        parts['cmeans'].set_linewidth(2)
        parts['cmeans'].set_color('red')
    if 'cmedians' in parts:
        parts['cmedians'].set_linestyle('-')
        parts['cmedians'].set_linewidth(2)
        parts['cmedians'].set_color('black')

    # X-axis labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels(filenames, rotation=30, ha='right')
    ax.set_ylabel("NCC Peak Values")
    
    # Main title with stats
    ax.set_title(f"Comparison of NCC Peaks\nMethod: {method}, p={p_value:.6f}, Effect Size: {effect_size:.6f}")
    
    # Annotate each violin with the total number of points
    data_list = [data1, data2]
    for i, data in enumerate(data_list):
        x_loc = i + 1
        y_max = np.max(data)
        y_min = np.min(data)
        # Add a small offset so the text sits above the top of the violin
        offset = 0.05 * (y_max - y_min) if (y_max - y_min) != 0 else 0.1
        y_pos = y_max + offset
        ax.text(x_loc, y_pos, f"N={len(data)}", ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    out_png = os.path.join(output_dir, "ncc_violin_plot.png")
    plt.savefig(out_png, dpi=150, compress_level=9)
    plt.close()
    logging.info(f"Saved violin plot to {out_png}")


def process_image_template(args_tuple):
    """
    Worker function to process one image-template pair.
    Returns: (filename, image_idx, template_idx, peak_coords, peak_values)
    """
    try:
        (filename, image_idx, template_idx, image, template, npeaks, diameter) = args_tuple
        ncc = compute_ncc_map_3d_single(image, template)
        peak_values, peak_coords = extract_ncc_peaks_with_diameter(ncc, npeaks, diameter)
        return (filename, image_idx, template_idx, peak_coords, peak_values)
    except Exception as e:
        logging.error("Error processing file {} image {} template {}: {}".format(
            filename, image_idx, template_idx, str(e)))
        logging.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Compute 3D NCC maps and compare peak distributions.")
    parser.add_argument('--input', required=True, help="Comma-separated input file paths.")
    parser.add_argument('--template', required=True, help="Template image file path (can be an HDF stack of templates).")
    parser.add_argument('--npeaks', type=int, default=10, help="Number of top peaks to extract per image-template pair.")
    parser.add_argument('--diameter', type=float, default=0,
                        help="Minimum allowed distance between peaks (in pixels). "
                             "If <=0, derive automatically from template’s bounding box.")
    parser.add_argument('--threads', type=int, default=1, help="Number of threads (processes) to use for parallel processing.")
    parser.add_argument('--output_dir', default="ncc_analysis", help="Output directory base name.")
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    setup_logging(output_dir)
    
    start_time = time.time()
    logging.info("Starting processing...")
    
    # Load template(s)
    templates = load_image(args.template).squeeze()
    if templates.ndim == 3:
        templates = np.expand_dims(templates, axis=0)
    n_templates = templates.shape[0]
    n_digits = len(str(n_templates))
    logging.info(f"Loaded {n_templates} template(s) from {args.template}")
    
    # Split and load the first tomogram to get shape for diameter clamping
    input_files = args.input.split(',')
    first_tomo_file = input_files[0]
    first_image_stack = load_image(first_tomo_file)
    if first_image_stack.ndim == 3:
        first_image_stack = np.expand_dims(first_image_stack, axis=0)
    first_tomo_shape = first_image_stack[0].shape
    
    # If user didn't provide a positive diameter, compute automatically from the first template & first tomogram
    if args.diameter <= 0:
        auto_diameter = compute_auto_diameter(templates[0], first_tomo_shape)
        args.diameter = auto_diameter
        logging.info(f"No diameter provided; using auto-derived diameter: {auto_diameter}")

    peak_values_by_file = defaultdict(list)
    tasks = []

    # Now proceed to create tasks for *all* input files
    for file_path in input_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            image_stack = load_image(file_path)
        except Exception as e:
            logging.error(f"Error loading image file {file_path}: {e}")
            continue
        if image_stack.ndim == 3:
            image_stack = np.expand_dims(image_stack, axis=0)
        n_images = image_stack.shape[0]
        for img_idx in range(n_images):
            for tmpl_idx in range(n_templates):
                tasks.append((filename, img_idx, tmpl_idx,
                              image_stack[img_idx], templates[tmpl_idx],
                              args.npeaks, args.diameter))

    results = []
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        future_to_task = {executor.submit(process_image_template, task): task for task in tasks}
        for future in as_completed(future_to_task):
            result = future.result()
            if result is not None:
                results.append(result)
    
    for (filename, image_idx, template_idx, peak_coords, peak_values) in results:
        template_str = str(template_idx).zfill(n_digits)
        save_peak_data(output_dir, filename, template_str, image_idx, peak_coords, peak_values, n_templates)
        plot_3d_peak_coordinates(output_dir, filename, template_str, image_idx, peak_coords)
        peak_values_by_file[filename].extend(peak_values.tolist())
    
    # If exactly two input files, produce a violin plot for quick comparison
    if len(peak_values_by_file) == 2:
        file_keys = list(peak_values_by_file.keys())
        data1 = np.array(peak_values_by_file[file_keys[0]])
        data2 = np.array(peak_values_by_file[file_keys[1]])
        plot_violin(data1, data2, file_keys, output_dir)
    
    elapsed = time.time() - start_time
    logging.info(f"Processing finished successfully in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Fatal error: " + str(e))
        logging.error(traceback.format_exc())
        sys.exit(1)
'''




'''PRESUMABLY WORKS MARCH 20 2025
#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 2025; last modification: Mar/13/2025
#
# Modifications:
#  1. Added logging and command-line logging to a log file.
#  2. Renamed 'coordinate_files' to 'coordinate_plots' and saved PNGs with compression.
#  3. Added support for multiple templates as an HDF stack.
#  4. Added --diameter filtering to ensure peaks are not too close.
#  5. Parallelized the NCC computation over image-template pairs using --threads.
#
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

def create_output_directory(base_dir):
    """Create and return a numbered output directory with needed subfolders."""
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "correlation_files"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "coordinate_plots"), exist_ok=True)
    return output_dir

def setup_logging(output_dir):
    """Set up logging to a file and console."""
    log_file = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Log the full command used to run the script.
    logging.info("Command: " + " ".join(sys.argv))
    return log_file

def load_image(file_path):
    """Loads a 3D image or a stack of 3D images from an HDF5 or MRC file."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.hdf', '.h5']:
        with h5py.File(file_path, 'r') as file:
            # Assumes structure MDF/images/<i>/image
            dataset_paths = [f"MDF/images/{i}/image" for i in range(len(file["MDF/images"]))]
            image_stack = np.array([file[ds][:] for ds in dataset_paths])
    elif ext == '.mrc':
        with mrcfile.open(file_path, permissive=True) as mrc:
            image_stack = mrc.data
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return image_stack

def compute_ncc_map_3d_single(image, template):
    """Computes the 3D normalized cross-correlation map for one image and one template."""
    if image.shape != template.shape:
        raise ValueError(f"Dimension mismatch: Image shape {image.shape}, Template shape {template.shape}")
    ncc = correlate(image, template, mode='same', method='fft')
    ncc = (ncc - np.mean(ncc)) / np.std(ncc)
    return ncc

def extract_ncc_peaks_with_diameter(ncc_map, npeaks, diameter):
    """
    Extracts up to npeaks from the NCC map ensuring that no two accepted peaks
    are closer than 'diameter'. Peaks are taken in descending order of NCC value.
    """
    sorted_indices = np.argsort(ncc_map.ravel())[::-1]
    accepted_coords = []
    accepted_values = []
    for idx in sorted_indices:
        coord = np.array(np.unravel_index(idx, ncc_map.shape))
        value = ncc_map.ravel()[idx]
        # If diameter filtering is enabled, check distances to already accepted peaks
        if diameter > 0 and accepted_coords:
            dists = np.linalg.norm(np.array(accepted_coords) - coord, axis=1)
            if np.any(dists < diameter):
                continue
        accepted_coords.append(coord)
        accepted_values.append(value)
        if len(accepted_coords) >= npeaks:
            break
    return np.array(accepted_values), np.array(accepted_coords)

def save_peak_data(output_dir, filename, template_str, image_idx, peak_coords, peak_values, total_templates):
    """
    Saves the peak coordinates and NCC values to a formatted text file.
    The output file will have columns: Template, X, Y, Z, NCC_Value.
    """
    # File name using the base filename, template and image indices
    image_str = str(image_idx).zfill(3)
    out_file = os.path.join(output_dir, 'correlation_files', f'ccc_{filename}_T{template_str}_I{image_str}.txt')
    header = "Template  X  Y  Z  NCC_Value"
    # Include the template number (already formatted with zfill)
    data = np.column_stack((np.full(peak_coords.shape[0], template_str), peak_coords, peak_values))
    fmt = '%s %5d %5d %5d %10.6f'
    np.savetxt(out_file, data, fmt=fmt, header=header)
    logging.info(f"Saved NCC peaks to {out_file}")

def plot_3d_peak_coordinates(output_dir, filename, template_str, image_idx, peak_coords):
    """Plots the NCC peak coordinates in 3D and saves the plot as a compressed PNG."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2], c='red', marker='o')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Top NCC Peak Locations")
    image_str = str(image_idx).zfill(3)
    out_png = os.path.join(output_dir, 'coordinate_plots', f'ccc_{filename}_T{template_str}_I{image_str}.png')
    # Save with compression settings (adjust dpi and compress_level as needed)
    plt.savefig(out_png, dpi=150, compress_level=9)
    plt.close()
    logging.info(f"Saved 3D peak plot to {out_png}")

def is_normal(data, alpha=0.05):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    _, p_value = shapiro(data)
    return p_value > alpha

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
    """Generates a violin plot comparing NCC peak distributions with relevant statistics."""
    p_value, effect_size, method = calculate_effect_size(data1, data2)
    fig, ax = plt.subplots(figsize=(8, 8))
    parts = ax.violinplot([data1, data2], showmeans=True, showmedians=True)

    # Customize colors
    colors = ['blue', 'orange']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
    if 'cmeans' in parts:
        parts['cmeans'].set_linestyle('--')
        parts['cmeans'].set_linewidth(2)
        parts['cmeans'].set_color('red')
    if 'cmedians' in parts:
        parts['cmedians'].set_linestyle('-')
        parts['cmedians'].set_linewidth(2)
        parts['cmedians'].set_color('black')
        
    ax.set_xticks([1, 2])
    ax.set_xticklabels(filenames, rotation=30, ha='right')
    ax.set_ylabel("NCC Peak Values")
    ax.set_title(f"Comparison of NCC Peaks\nMethod: {method}, p={p_value:.6f}, Effect Size: {effect_size:.6f}")
    plt.tight_layout()
    out_png = os.path.join(output_dir, "ncc_violin_plot.png")
    plt.savefig(out_png, dpi=150, compress_level=9)
    plt.close()
    logging.info(f"Saved violin plot to {out_png}")

def process_image_template(args_tuple):
    """
    Worker function to process one image-template pair.
    Expects a tuple:
      (filename, image_idx, template_idx, image, template, npeaks, diameter)
    Returns: (filename, image_idx, template_idx, peak_coords, peak_values)
    """
    try:
        (filename, image_idx, template_idx, image, template, npeaks, diameter) = args_tuple
        # Compute NCC map
        ncc = compute_ncc_map_3d_single(image, template)
        # Extract peaks with diameter filtering
        peak_values, peak_coords = extract_ncc_peaks_with_diameter(ncc, npeaks, diameter)
        return (filename, image_idx, template_idx, peak_coords, peak_values)
    except Exception as e:
        logging.error("Error processing file {} image {} template {}: {}".format(filename, image_idx, template_idx, str(e)))
        logging.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Compute 3D NCC maps and compare peak distributions.")
    parser.add_argument('--input', required=True, help="Comma-separated input file paths.")
    parser.add_argument('--template', required=True, help="Template image file path (can be an HDF stack of templates).")
    parser.add_argument('--npeaks', type=int, default=10, help="Number of top peaks to extract per image-template pair.")
    parser.add_argument('--diameter', type=float, default=0, help="Minimum allowed distance between peaks (in pixels); 0 means no filtering.")
    parser.add_argument('--threads', type=int, default=1, help="Number of threads (processes) to use for parallel processing.")
    parser.add_argument('--output_dir', default="ncc_analysis", help="Output directory base name.")
    args = parser.parse_args()
    
    # Create output directory and setup logging.
    output_dir = create_output_directory(args.output_dir)
    setup_logging(output_dir)
    
    start_time = time.time()
    logging.info("Starting processing...")
    
    # Load template(s)
    templates = load_image(args.template).squeeze()
    # Ensure templates is 4D: (n_templates, x, y, z)
    if templates.ndim == 3:
        templates = np.expand_dims(templates, axis=0)
    n_templates = templates.shape[0]
    n_digits = len(str(n_templates))
    logging.info(f"Loaded {n_templates} template(s) from {args.template}")
    
    input_files = args.input.split(',')
    # Dictionary to accumulate peak values for violin plot (if exactly two input files)
    peak_values_by_file = defaultdict(list)
    
    # Prepare a list of tasks (each one is a tuple for an image-template pair)
    tasks = []
    for file_path in input_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            image_stack = load_image(file_path)
        except Exception as e:
            logging.error(f"Error loading image file {file_path}: {e}")
            continue
        # If image_stack is 3D then assume one image
        if image_stack.ndim == 3:
            image_stack = np.expand_dims(image_stack, axis=0)
        n_images = image_stack.shape[0]
        for img_idx in range(n_images):
            for tmpl_idx in range(n_templates):
                tasks.append((filename, img_idx, tmpl_idx, image_stack[img_idx], templates[tmpl_idx],
                              args.npeaks, args.diameter))
    
    # Process tasks in parallel using ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        future_to_task = {executor.submit(process_image_template, task): task for task in tasks}
        for future in as_completed(future_to_task):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Process the results: Save peak data and generate plots.
    for (filename, image_idx, template_idx, peak_coords, peak_values) in results:
        # Format the template number string with zfill.
        template_str = str(template_idx).zfill(n_digits)
        save_peak_data(output_dir, filename, template_str, image_idx, peak_coords, peak_values, n_templates)
        plot_3d_peak_coordinates(output_dir, filename, template_str, image_idx, peak_coords)
        # Accumulate peak values for violin plot (grouped by input file)
        peak_values_by_file[filename].extend(peak_values.tolist())
    
    # Optionally, if exactly two input files were processed, generate a violin plot.
    if len(peak_values_by_file) == 2:
        file_keys = list(peak_values_by_file.keys())
        data1 = np.array(peak_values_by_file[file_keys[0]])
        data2 = np.array(peak_values_by_file[file_keys[1]])
        plot_violin(data1, data2, file_keys, output_dir)
    
    elapsed = time.time() - start_time
    logging.info(f"Processing finished successfully in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Fatal error: " + str(e))
        logging.error(traceback.format_exc())
        sys.exit(1)
'''
















'''
WORKS IN FEBRUARY 2025

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

    fig, ax = plt.subplots(figsize=(8, 8))
    parts = ax.violinplot([data1, data2], showmeans=True, showmedians=True)

    # Customizing violin colors
    colors = ['blue', 'orange']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    # Ensure mean and median lines are clearly distinguishable
    if 'cmeans' in parts:
        parts['cmeans'].set_linestyle('--')  # Dashed style for means
        parts['cmeans'].set_linewidth(2)
        parts['cmeans'].set_color('red')
    
    if 'cmedians' in parts:
        parts['cmedians'].set_linestyle('-')  # Solid style for medians
        parts['cmedians'].set_linewidth(2)
        parts['cmedians'].set_color('black')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(filenames, rotation=30, ha='right')  # Prevent overlapping labels
    ax.set_ylabel("NCC Peak Values")
    ax.set_title(f"Comparison of NCC Peaks\nMethod: {method}, p={p_value:.6f}, Effect Size: {effect_size:.6f}")

    # Adjust statistics box placement dynamically to avoid overlapping the title
    max_y = max(max(data1), max(data2))
    min_y = min(min(data1), min(data2))
    y_offset = (max_y - min_y) * 0.2  # Increase offset for better spacing

    for i, data in enumerate([data1, data2]):
        mean_val, median_val = np.mean(data), np.median(data)
        ax.text(i + 1, max_y + y_offset, 
                f"N={len(data)}\nGaussian: {is_normal(data)}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}",
                ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ncc_violin_plot.png"))
    plt.close()

    

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










'''
INITIAL VERSION

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

