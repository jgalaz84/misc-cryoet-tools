#!/usr/bin/env python
"""
# Author: Jesus Galaz-Montoya 03/2025 (last modified 03/2025)
#
# tomo_eval_edges.py
#
# Description:

This script reads 3D tomogram stacks from HDF or MRC files and evaluates
their edges. For each 3D volume it checks eight corners and/or six faces
to determine if any of these regions are “empty” (i.e. if the region’s
mean, standard deviation, minimum, and maximum are all zero).

Parameters controlling the probing:
  --corner_size: Side length (in pixels) of the cubic region to extract from
                 each corner. Default is 3. If set to 0, corner evaluation is disabled.
  --slice_thickness: Thickness (in pixels) of the slab to extract from each face.
                     Default is 1. If set to 0, face evaluation is disabled.

Output:
  For each HDF file, if any volumes have empty edge regions, two text files are produced:
    • {root}_exclude.txt – a single column (one volume index per line) listing
      volumes to exclude.
    • {root}_exclude_reasons.txt – each line lists a volume index and (separated by tabs)
      which corners (e.g. x0,y0,z0, x0,y0,z1, ...) or faces (yx0, yx1, xz0, xz1, yz0, yz1)
      were empty. Here, "0" is "left" and "1" is "right" for x, y, and z axes, to indicate
      the corner or face positions.
      
  For MRC files (which are single-volume), only a {root}_exclude_reasons.txt file is produced,
  listing the filename and reasons.

  If --save_clean_stack is provided, a new HDF stack (or, for MRC files, symlinks in a
  subdirectory "clean_mrcs") is created containing only volumes that do not have empty edge regions.

If no volumes (or MRC images) are found to have empty corners/faces, no output files are written.
Basic progress and result messages (such as "No volumes with empty corners or edge slices found")
are printed regardless of the verbosity level.

Usage example:
  ./tomo_eval_edges.py --input mydata --corner_size 3 --slice_thickness 1 --output_dir tomo_eval_edges --save_clean_stack --verbose 2
"""

import argparse
import os
import sys
import glob
import numpy as np
import h5py
import mrcfile
from EMAN2 import EMNumPy, EMData
import numpy as np
import os

# --------------------------
# Helper functions
# --------------------------

def create_output_directory(base_dir):
    """Create and return a unique, numbered output directory."""
    i = 0
    while os.path.exists(f"{base_dir}_{i:02}"):
        i += 1
    output_dir = f"{base_dir}_{i:02}"
    os.makedirs(output_dir)
    return output_dir

def is_empty(region):
    """Return True if the region (a numpy array) is entirely zero (within tolerance)."""
    return np.allclose(region, 0)

def check_corners(volume, corner_size):
    """
    Check the 8 corners of a 3D volume (assumed shape (z,y,x)).
    Returns a dict mapping corner labels (e.g. "x0,y0,z0") to booleans (True if empty).
    """
    try:
        z_dim, y_dim, x_dim = volume.shape
    except ValueError:
        raise ValueError("Volume is not 3D. Got shape: " + str(volume.shape))
    cs = corner_size if corner_size <= min(z_dim, y_dim, x_dim) else min(z_dim, y_dim, x_dim)
    indices = {
         "x0": 0, "x1": x_dim - cs,
         "y0": 0, "y1": y_dim - cs,
         "z0": 0, "z1": z_dim - cs
    }
    corners = {
        "x0,y0,z0": volume[indices["z0"]:indices["z0"]+cs, indices["y0"]:indices["y0"]+cs, indices["x0"]:indices["x0"]+cs],
        "x0,y0,z1": volume[indices["z1"]:indices["z1"]+cs, indices["y0"]:indices["y0"]+cs, indices["x0"]:indices["x0"]+cs],
        "x0,y1,z0": volume[indices["z0"]:indices["z0"]+cs, indices["y1"]:indices["y1"]+cs, indices["x0"]:indices["x0"]+cs],
        "x0,y1,z1": volume[indices["z1"]:indices["z1"]+cs, indices["y1"]:indices["y1"]+cs, indices["x0"]:indices["x0"]+cs],
        "x1,y0,z0": volume[indices["z0"]:indices["z0"]+cs, indices["y0"]:indices["y0"]+cs, indices["x1"]:indices["x1"]+cs],
        "x1,y0,z1": volume[indices["z1"]:indices["z1"]+cs, indices["y0"]:indices["y0"]+cs, indices["x1"]:indices["x1"]+cs],
        "x1,y1,z0": volume[indices["z0"]:indices["z0"]+cs, indices["y1"]:indices["y1"]+cs, indices["x1"]:indices["x1"]+cs],
        "x1,y1,z1": volume[indices["z1"]:indices["z1"]+cs, indices["y1"]:indices["y1"]+cs, indices["x1"]:indices["x1"]+cs],
    }
    results = {}
    for label, region in corners.items():
         results[label] = is_empty(region)
    return results

def check_faces(volume, slice_thickness):
    """
    Check the 6 faces of a 3D volume (assumed shape (z,y,x)).
    Returns a dict mapping face labels to booleans (True if empty).

    Face labels:
      yx0: face at z=0, in the y-x plane.
      yx1: face at z=end.
      xz0: face at y=0, in the x-z plane.
      xz1: face at y=end.
      yz0: face at x=0, in the y-z plane.
      yz1: face at x=end.
    """
    try:
        z_dim, y_dim, x_dim = volume.shape
    except ValueError:
        raise ValueError("Volume is not 3D. Got shape: " + str(volume.shape))
    st = slice_thickness if slice_thickness <= min(z_dim, y_dim, x_dim) else min(z_dim, y_dim, x_dim)
    results = {}
    results["yx0"] = is_empty(volume[0:st, :, :])
    results["yx1"] = is_empty(volume[z_dim-st:z_dim, :, :])
    results["xz0"] = is_empty(volume[:, 0:st, :])
    results["xz1"] = is_empty(volume[:, y_dim-st:y_dim, :])
    results["yz0"] = is_empty(volume[:, :, 0:st])
    results["yz1"] = is_empty(volume[:, :, x_dim-st:x_dim])
    return results

# --------------------------
# Processing functions for HDF and MRC
# --------------------------

def process_hdf_file(file_path, corner_size, slice_thickness, save_clean_stack, verbose):
    """
    Process a multi-volume HDF file.

    Returns:
      excluded_indices: list of global (0-based) volume indices with empty edges.
      reasons_dict: dict mapping each excluded index to a tab-delimited string of empty region labels.
      clean_volumes: numpy array of volumes that passed the test (if save_clean_stack is True), or None.
    """
    if verbose >= 2:
         print(f"Opening HDF file: {file_path}")
    excluded_indices = []
    reasons_dict = {}
    clean_volumes = []
    with h5py.File(file_path, 'r') as f:
         paths = []
         def visitor(name, node):
              if isinstance(node, h5py.Dataset) and name.endswith('/image'):
                   paths.append(name)
         f.visititems(visitor)
         if not paths:
              raise KeyError("No datasets with '/image' found in the file.")
         if verbose >= 2:
              print(f"Found datasets: {paths}")
         global_idx = 0  # global volume index across all datasets
         for dset in paths:
              data = f[dset][:]
              # Determine if the dataset is a single volume (3D) or a stack (4D or higher)
              if data.ndim == 3:
                   volumes = [data]
                   if verbose >= 2:
                        print(f"Dataset {dset} is 3D; treating as a single volume.")
              elif data.ndim >= 4:
                   volumes = [data[i] for i in range(data.shape[0])]
                   if verbose >= 2:
                        print(f"Dataset {dset} is {data.ndim}D; processing {data.shape[0]} volumes.")
              else:
                   if verbose >= 1:
                        print(f"Skipping dataset {dset} with unsupported ndim {data.ndim}")
                   continue
              for idx, volume in enumerate(volumes):
                    if verbose >= 2:
                         print(f"Processing volume {global_idx} (dataset {dset}, local index {idx})")
                    reasons = []
                    if corner_size > 0:
                         try:
                             corners = check_corners(volume, corner_size)
                         except Exception as e:
                             if verbose >= 1:
                                  print(f"Error checking corners for volume {global_idx}: {e}")
                             continue
                         for label, empty in corners.items():
                              if empty:
                                  reasons.append(label)
                    if slice_thickness > 0:
                         try:
                             faces = check_faces(volume, slice_thickness)
                         except Exception as e:
                             if verbose >= 1:
                                  print(f"Error checking faces for volume {global_idx}: {e}")
                             continue
                         for label, empty in faces.items():
                              if empty:
                                  reasons.append(label)
                    if reasons:
                         excluded_indices.append(global_idx)
                         reasons_dict[global_idx] = "\t".join(reasons)
                         if verbose >= 2:
                              print(f"Volume {global_idx} excluded for reasons: {reasons}")
                    else:
                         clean_volumes.append(volume)
                         if verbose >= 2:
                              print(f"Volume {global_idx} is clean.")
                    global_idx += 1
         clean_volumes = np.array(clean_volumes) if clean_volumes else None
         return excluded_indices, reasons_dict, clean_volumes

def process_mrc_file(file_path, corner_size, slice_thickness, verbose):
    """
    Process a single-volume MRC file.
    
    Returns a list of empty-region labels (if any). An empty list means the volume is clean.
    """
    if verbose >= 2:
         print(f"Opening MRC file: {file_path}")
    with mrcfile.open(file_path, permissive=True) as mrc:
         volume = mrc.data
    reasons = []
    if corner_size > 0:
         try:
             corners = check_corners(volume, corner_size)
         except Exception as e:
             if verbose >= 1:
                  print(f"Error checking corners for MRC {file_path}: {e}")
             corners = {}
         for label, empty in corners.items():
              if empty:
                  reasons.append(label)
    if slice_thickness > 0:
         try:
             faces = check_faces(volume, slice_thickness)
         except Exception as e:
             if verbose >= 1:
                  print(f"Error checking faces for MRC {file_path}: {e}")
             faces = {}
         for label, empty in faces.items():
              if empty:
                  reasons.append(label)
    if verbose >= 2:
         if reasons:
              print(f"MRC {file_path} excluded for reasons: {reasons}")
         else:
              print(f"MRC {file_path} is clean.")
    return reasons



def save_clean_hdf_stack(clean_volumes, output_path, verbose):
    """
    Write an EMAN2-compatible HDF stack of 3D volumes.
    Each volume is appended to the file at a new index.
    e2iminfo.py and e2display.py should recognize this output.
    """
    # If a file with the same name exists, remove it so we start fresh
    if os.path.exists(output_path):
        os.remove(output_path)

    # For each volume in 'clean_volumes':
    for i, vol in enumerate(clean_volumes):
        # Ensure the data is float32 (EMAN2 typically uses single-precision floats)
        vol_f32 = vol.astype(np.float32, copy=False)
        
        # Convert this NumPy array into an EMData object
        emdata_obj = EMNumPy.numpy2em(vol_f32)

        # Write to the output HDF file at image index i
        #   - If i=0, it creates a new file
        #   - If i>0, it appends
        emdata_obj.write_image(output_path, i)

    if verbose >= 1:
        print(f"Wrote {len(clean_volumes)} clean volumes to EMAN2 HDF stack: {output_path}")



# --------------------------
# Main function
# --------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate edges of 3D tomograms to identify empty regions in corners and faces.\n"
                    "  --corner_size defines the side length (in pixels) of the cubic region to probe at each corner (default=3; 0 disables corner testing).\n"
                    "  --slice_thickness defines the thickness (in pixels) of the slab to probe at each face (default=1; 0 disables face testing).")
    parser.add_argument('--input', required=True,
                        help="Input file name or common string to search for (supports .hdf, .h5, .mrc).")
    parser.add_argument('--output_dir', default="tomo_eval_edges",
                        help="Base output directory (the script will create a numbered directory, e.g., tomo_eval_edges_00).")
    parser.add_argument('--corner_size', type=int, default=3,
                        help="Side length (in pixels) of the cubic region to probe at each corner. 0 disables corner testing.")
    parser.add_argument('--slice_thickness', type=int, default=1,
                        help="Thickness (in pixels) of the slab to probe on each face. 0 disables face testing.")
    parser.add_argument('--save_clean_stack', action='store_true',
                        help="If set, save a new HDF stack (or symlink clean MRCs) containing only volumes that do not have empty regions.")
    parser.add_argument('--verbose', type=int, default=1,
                        help="Verbosity level: 0 = silent, 1 = essential output, 2 = detailed progress messages.")
    args = parser.parse_args()
    
    verbose = args.verbose
    
    # Determine input files: if the --input string has a valid extension, use it; otherwise, search for matching files.
    valid_extensions = ['.hdf', '.h5', '.mrc']
    if os.path.splitext(args.input)[1].lower() in valid_extensions:
         input_files = [args.input]
    else:
         input_files = []
         for ext in valid_extensions:
              pattern = f"*{args.input}*{ext}"
              input_files.extend(glob.glob(pattern))
         input_files = list(set(input_files))
         if not input_files:
              print("No files found matching the input string.")
              sys.exit(1)
    
    out_dir = create_output_directory(args.output_dir)
    print(f"Output directory: {out_dir}")
    
    # For saving clean MRC symlinks if needed.
    clean_mrc_dir = None
    if args.save_clean_stack and any(f.lower().endswith('.mrc') for f in input_files):
         clean_mrc_dir = os.path.join(out_dir, "clean_mrcs")
         os.makedirs(clean_mrc_dir, exist_ok=True)
         if verbose >= 2:
              print(f"Clean MRC symlink directory: {clean_mrc_dir}")
    
    # Process each file
    for file_path in input_files:
         base = os.path.splitext(os.path.basename(file_path))[0]
         ext = os.path.splitext(file_path)[1].lower()
         print(f"Processing file: {file_path}")
         if ext in ['.hdf', '.h5']:
              try:
                   excluded, reasons, clean_vols = process_hdf_file(file_path, args.corner_size, args.slice_thickness, args.save_clean_stack, verbose)
              except Exception as e:
                   print(f"Error processing {file_path}: {e}")
                   continue
              if excluded:
                   print(f"Found {len(excluded)} volumes with empty corners/edges in {file_path}.")
                   # Write output files for HDF: one with indices and one with reasons.
                   exclude_txt = os.path.join(out_dir, f"{base}_exclude.txt")
                   exclude_reasons_txt = os.path.join(out_dir, f"{base}_exclude_reasons.txt")
                   with open(exclude_txt, 'w') as f:
                        for idx in excluded:
                             f.write(f"{idx}\n")
                   with open(exclude_reasons_txt, 'w') as f:
                        for idx in sorted(reasons.keys()):
                             f.write(f"{idx}\t{reasons[idx]}\n")
                   print(f"Saved: {exclude_txt} and {exclude_reasons_txt}")
                   if args.save_clean_stack and clean_vols is not None and clean_vols.size > 0:
                        clean_hdf_path = os.path.join(out_dir, f"{base}_clean.hdf")
                        save_clean_hdf_stack(clean_vols, clean_hdf_path, verbose)
              else:
                   print(f"No volumes with empty corners or edge slices found in {file_path}. No output files generated.")
         elif ext == '.mrc':
              reasons = process_mrc_file(file_path, args.corner_size, args.slice_thickness, verbose)
              if reasons:
                   print(f"Found empty regions in MRC {file_path}: {reasons}")
                   exclude_reasons_txt = os.path.join(out_dir, f"{base}_exclude_reasons.txt")
                   with open(exclude_reasons_txt, 'w') as f:
                        f.write(f"{os.path.basename(file_path)}\t" + "\t".join(reasons) + "\n")
                   print(f"Saved: {exclude_reasons_txt}")
                   # For clean stack: do not create symlink if image is excluded.
              else:
                   print(f"No empty regions found in MRC {file_path}. No output file generated.")
                   if args.save_clean_stack and clean_mrc_dir is not None:
                        # Create symlink only if the file is clean.
                        link_name = os.path.join(clean_mrc_dir, os.path.basename(file_path))
                        try:
                             if not os.path.exists(link_name):
                                  os.symlink(os.path.abspath(file_path), link_name)
                             print(f"Created symlink for clean MRC: {link_name}")
                        except Exception as e:
                             print(f"Could not create symlink for {file_path}: {e}")
         else:
              print(f"Unsupported file extension: {ext}")

if __name__ == '__main__':
    main()
