#!/usr/bin/env python
"""
tomo_sim_pdb.py

This program requires havinv EMAN2 installed. 
It generates a simulated "ground truth" tomogram by placing copies of 3D models (from local files,
EMDB, or PDB) at random positions and orientations in a larger volume.

This version implements an elaborate packing algorithm that considers each species’ own size.
Each species’ “radius” (half of its longest span computed from a bounding box of significant density)
is used so that a new particle (with radius r_new) can only be placed if its center is at least 
r_existing + r_new away from every previously placed particle. In equimolar mode (--equimolar),
the algorithm cycles through species in round-robin fashion so that roughly equal numbers are placed.
Otherwise, species are processed in descending order of radius (so larger molecules are placed first).

Additional features:
  - Grid positions are determined by random sampling with an occupancy map.
  - The remainder of the volume is equally distributed at both ends in each dimension so that placements are centered.
  - Output is in MRC format by default (unless --hdfoutput is supplied).
  - Verbose feedback (controlled by --verbose) and logging (with E2init/E2end) are provided.
  
Author: Jesus G. Galaz-Montoya
Date: Oct 2024
Last Update: Mar 2025
"""

import os
import sys
import math
import random
import argparse
import time
import tempfile
import shutil
import gzip
import numpy as np
from datetime import timedelta

# Import EMAN2 modules
from EMAN2 import EMData, EMUtil, Transform, EMArgumentParser, E2init, E2end
import requests

##############################################################################
# Helper functions for output directory and grid packing
##############################################################################
def make_output_dir(base, verbose=0):
    """Create a new numbered output directory (e.g., base_00, base_01, …)"""
    suffix = 0
    while True:
        dirname = f"{base}_{suffix:02d}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            if verbose > 0:
                print(f"[Verbose >0] Created output directory: {dirname}")
            return dirname
        suffix += 1

def compute_grid_positions(length, spacing):
    """
    Compute grid positions along a dimension of length 'length' with cells of size 'spacing'.
    The remainder is distributed equally at both ends so that the positions are centered.
    Returns a list of center coordinates.
    """
    if length < spacing:
        return [int(round(length/2.0))]
    N = int(length // spacing)
    remainder = length - (N * spacing)
    margin = remainder / 2.0
    positions = []
    for i in range(N):
        pos = margin + spacing/2.0 + i * spacing
        positions.append(int(round(pos)))
    return positions

def get_coords(options, spacing):
    """
    Generate grid coordinates for the volume using the computed spacing.
    This uses compute_grid_positions to center the grid in each dimension.
    """
    xs = compute_grid_positions(options.tomonx, spacing)
    ys = compute_grid_positions(options.tomony, spacing)
    zs = compute_grid_positions(options.tomonz, spacing)
    coords = []
    for x in xs:
        for y in ys:
            for z in zs:
                coords.append((x, y, z))
    return coords

##############################################################################
# Functions to compute particle size and create spherical masks
##############################################################################
def calculate_structure_diagonal(model, threshold=0.05):
    """
    Calculate the diagonal (i.e. longest span) of the structure in 'model'.
    This finds the bounding box of voxels with density > threshold and returns the Euclidean distance.
    If no voxel exceeds the threshold, returns model["nx"].
    """
    data = model.numpy()
    indices = np.nonzero(data > threshold)
    if indices[0].size == 0:
        return model["nx"]
    x_min, x_max = indices[0].min(), indices[0].max()
    y_min, y_max = indices[1].min(), indices[1].max()
    z_min, z_max = indices[2].min(), indices[2].max()
    diag = math.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)
    return diag

def get_sphere_mask(radius):
    """
    Compute and return a list of (dx,dy,dz) offsets for a sphere of given radius.
    Voxels with (dx^2 + dy^2 + dz^2) <= radius^2 are included.
    """
    r_int = int(math.ceil(radius))
    offsets = []
    for dx in range(-r_int, r_int+1):
        for dy in range(-r_int, r_int+1):
            for dz in range(-r_int, r_int+1):
                if dx*dx + dy*dy + dz*dz <= radius*radius:
                    offsets.append((dx, dy, dz))
    return offsets

##############################################################################
# Normalization function
##############################################################################
def normalize_model(model, options):
    """
    Normalize the model:
      1) Shift the data so that the minimum becomes 0.
      2) Scale the data so that the maximum becomes 1.
      3) Binarize: set voxels to 1 if above (mean + threshold_sigma*std), else 0.
    Returns a new EMData object containing the binary data.
    """
    # Access the data as a NumPy array.
    data = model.numpy()
    # Shift the data.
    min_val = data.min()
    data = data - min_val
    # Scale the data.
    max_val = data.max()
    if max_val != 0:
        data = data / max_val
    # Compute threshold.
    thresh = data.mean() + options.threshold_sigma * data.std()
    bin_data = (data >= thresh).astype(np.float32)
    nx, ny, nz = bin_data.shape
    new_model = EMData(nx, ny, nz)
    new_model.set_data_string(bin_data.tobytes())
    new_model.set_attr_dict({"minimum": float(bin_data.min()),
                             "maximum": float(bin_data.max())})
    # Propagate the source attribute.
    new_model.set_attr("source", model.get_attr_default("source", "unknown"))
    return new_model

##############################################################################
# Occupancy-based packing functions
##############################################################################
def can_place(occupancy, center, mask, volume_dims):
    """
    Check if a particle can be placed at 'center' given its spherical mask.
    For each offset in mask, verify that (center+offset) is inside volume and unoccupied.
    Returns True if placement is possible.
    """
    X, Y, Z = volume_dims
    cx, cy, cz = center
    for (dx, dy, dz) in mask:
        x = cx + dx
        y = cy + dy
        z = cz + dz
        if x < 0 or x >= X or y < 0 or y >= Y or z < 0 or z >= Z:
            return False  # Out of bounds.
        if occupancy[x, y, z]:
            return False
    return True

def mark_occupancy(occupancy, center, mask, volume_dims):
    """
    Mark the voxels in the occupancy array as occupied for a particle placed at 'center'
    with a given spherical mask.
    """
    X, Y, Z = volume_dims
    cx, cy, cz = center
    for (dx, dy, dz) in mask:
        x = cx + dx
        y = cy + dy
        z = cz + dz
        if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
            occupancy[x, y, z] = True

def pack_particles(species_list, volume_dims, total_n, equimolar, max_attempts, verbose):
    """
    Pack particles into the volume using an occupancy-based algorithm.
    
    species_list: a list of dicts (one per species) with keys:
       "model"   : normalized EMData for that species.
       "radius"  : computed radius (half of longest span).
       "mask"    : list of voxel offsets for the spherical mask.
       "source"  : identifier string.
    volume_dims: tuple (X, Y, Z) of the tomogram dimensions.
    total_n: total number of particles to place.
    equimolar: if True, place particles in a round-robin fashion; else, sort by descending radius.
    max_attempts: maximum random attempts per species (or per round in equimolar mode).
    verbose: verbosity level.
    
    Returns a list of placements (dicts with keys "x", "y", "z", "species_idx").
    """
    X, Y, Z = volume_dims
    occupancy = np.zeros((X, Y, Z), dtype=bool)
    placements = []
    
    if equimolar:
        species_counts = [0] * len(species_list)
        total = 0
        while total < total_n:
            placed_in_round = False
            for i, species in enumerate(species_list):
                found = False
                for attempt in range(max_attempts):
                    x = random.randint(0, X-1)
                    y = random.randint(0, Y-1)
                    z = random.randint(0, Z-1)
                    if can_place(occupancy, (x, y, z), species["mask"], volume_dims):
                        mark_occupancy(occupancy, (x, y, z), species["mask"], volume_dims)
                        placements.append({"x": x, "y": y, "z": z, "species_idx": i})
                        species_counts[i] += 1
                        total += 1
                        found = True
                        if verbose > 2:
                            print(f"[Verbose >2] Placed species {species['source']} at ({x},{y},{z})")
                        break
                if found:
                    placed_in_round = True
            if not placed_in_round:
                if verbose > 0:
                    print("[Verbose >0] No placements possible in this round; stopping equimolar packing.")
                break
        return placements
    else:
        # Non-equimolar: sort species indices by descending radius.
        sorted_indices = sorted(range(len(species_list)), key=lambda i: species_list[i]["radius"], reverse=True)
        total = 0
        for i in sorted_indices:
            species = species_list[i]
            for attempt in range(max_attempts):
                if total >= total_n:
                    break
                x = random.randint(0, X-1)
                y = random.randint(0, Y-1)
                z = random.randint(0, Z-1)
                if can_place(occupancy, (x, y, z), species["mask"], volume_dims):
                    mark_occupancy(occupancy, (x, y, z), species["mask"], volume_dims)
                    placements.append({"x": x, "y": y, "z": z, "species_idx": i})
                    total += 1
                    if verbose > 2:
                        print(f"[Verbose >2] Placed species {species['source']} at ({x},{y},{z})")
            # Continue to next species even if not reaching total_n.
        return placements

##############################################################################
# Main function
##############################################################################
def main():
    start_time = time.perf_counter()
    
    # -------------------------------------------------------------------------
    # Parse command-line arguments.
    # -------------------------------------------------------------------------
    parser = EMArgumentParser(usage="tomo_sim_pdb.py [options]", version="1.0")
    parser.add_argument("--apix", type=float, default=1.0,
                        help="Target pixel size for simulated volume. Models will be rescaled if needed.")
    parser.add_argument("--tomonx", type=int, default=256, help="Tomogram size in x.")
    parser.add_argument("--tomony", type=int, default=256, help="Tomogram size in y.")
    parser.add_argument("--tomonz", type=int, default=168, help="Tomogram size in z.")
    parser.add_argument("--box_size", type=int, default=64,
                        help="Size (in voxels) for each particle's bounding box.")
    parser.add_argument("--dilutionfactor", type=int, default=1,
                        help="Dilution factor; 1 means maximum crowdedness, 2 means half as many particles, etc.")
    parser.add_argument("--nptcls", type=int, default=None,
                        help="Total number of particles to insert. If not supplied, a high default is used.")
    parser.add_argument("--input", type=str, default=None,
                        help="Comma-separated list of local volumes (MRC/HDF).")
    parser.add_argument("--input_emdb", type=str, default=None,
                        help="Comma-separated list of EMDB accession numbers.")
    parser.add_argument("--input_pdb", type=str, default=None,
                        help="Comma-separated list of PDB accession numbers.")
    parser.add_argument("--threshold_sigma", type=float, default=3.0,
                        help="Sigma threshold for binarizing (default=3.0).")
    parser.add_argument("--savesteps", action="store_true", default=False,
                        help="Save intermediate normalized models to disk.")
    parser.add_argument("--background", type=float, default=0.0,
                        help="Background density (e.g., 0.71428571428 for ice; 0 for black).")
    parser.add_argument("--output", type=str, default="simulated_tomogram.hdf",
                        help="Filename for the final tomogram (default HDF).")
    parser.add_argument("--output_label", type=str, default="simulated_labels.hdf",
                        help="Filename for the label (segmentation) map (default HDF).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory; if not given, a numbered directory (e.g., tomo_gt_pdb_00) is created.")
    parser.add_argument("--hdfoutput", action="store_true", default=False,
                        help="Output in HDF format; if not set, output will be in MRC format.")
    parser.add_argument("--equimolar", action="store_true", default=False,
                        help="If set, place an equal number of particles for each unique input.")
    parser.add_argument("--max_attempts", type=int, default=10000,
                        help="Maximum random attempts per species (or per round in equimolar mode).")
    parser.add_argument("--verbose", "-v", type=int, default=0,
                        help="Verbosity level [0..9]. Higher means more output.")
    
    (options, args) = parser.parse_args()
    
    # Force output to MRC unless --hdfoutput is set.
    if not options.hdfoutput:
        if not options.output.lower().endswith(".mrc"):
            options.output = os.path.splitext(options.output)[0] + ".mrc"
        if not options.output_label.lower().endswith(".mrc"):
            options.output_label = os.path.splitext(options.output_label)[0] + ".mrc"
    
    # Initialize the EMAN2 logger.
    logger = E2init(sys.argv, options.ppid if hasattr(options, "ppid") else -1)
    
    if options.verbose > 0:
        print(f"[Verbose >0] Starting simulation with apix={options.apix}, box_size={options.box_size}, "
              f"volume=({options.tomonx}, {options.tomony}, {options.tomonz})")
    
    # -------------------------------------------------------------------------
    # Create output directory.
    # -------------------------------------------------------------------------
    if options.output_dir is None:
        options.output_dir = make_output_dir("tomo_gt_pdb", verbose=options.verbose)
    else:
        if not os.path.exists(options.output_dir):
            os.makedirs(options.output_dir)
            if options.verbose > 0:
                print(f"[Verbose >0] Created output directory: {options.output_dir}")
    
    # -------------------------------------------------------------------------
    # Load input models (local, EMDB, PDB) and tag with source.
    # -------------------------------------------------------------------------
    models = []
    if options.input:
        file_list = [f.strip() for f in options.input.split(',') if f.strip()]
        for f in file_list:
            if options.verbose > 0:
                print(f"[Verbose >0] Loading local volume: {f}")
            try:
                n_images = EMUtil.get_image_count(f)
                if n_images > 1:
                    for i in range(n_images):
                        vol = EMData(f, i)
                        vol.set_attr("source", f)
                        models.append(vol)
                        if options.verbose > 1:
                            print(f"[Verbose >1]  Loaded sub-volume {i} from {f}")
                else:
                    vol = EMData(f, 0)
                    vol.set_attr("source", f)
                    models.append(vol)
            except Exception as e:
                sys.stderr.write(f"Error loading file {f}: {e}\n")
    if options.input_emdb:
        emdb_list = [s.strip() for s in options.input_emdb.split(',') if s.strip()]
        for acc in emdb_list:
            if options.verbose > 0:
                print(f"[Verbose >0] Fetching EMDB map: {acc}")
            vol = fetch_emdb_map(acc, options)
            if vol:
                models.append(vol)
    if options.input_pdb:
        pdb_list = [s.strip() for s in options.input_pdb.split(',') if s.strip()]
        for acc in pdb_list:
            if options.verbose > 0:
                print(f"[Verbose >0] Fetching PDB model: {acc}")
            vol = fetch_pdb_map(acc, options)
            if vol:
                models.append(vol)
    
    if len(models) == 0:
        sys.stderr.write("No models were loaded. Exiting.\n")
        sys.exit(1)
    if options.verbose > 0:
        print(f"[Verbose >0] Loaded {len(models)} model(s).")
    
    # -------------------------------------------------------------------------
    # Determine target apix and rescale models if needed.
    # -------------------------------------------------------------------------
    target_apix = options.apix
    for m in models:
        apix_val = m.get_attr_default("apix_x", options.apix)
        if apix_val > target_apix:
            target_apix = apix_val
    if options.verbose > 0:
        print(f"[Verbose >0] Final target apix = {target_apix}")
    
    models_rescaled = []
    for i, m in enumerate(models):
        apix_val = m.get_attr_default("apix_x", options.apix)
        if abs(apix_val - target_apix) > 1e-5:
            scale_factor = apix_val / target_apix
            if options.verbose > 1:
                print(f"[Verbose >1] Rescaling model {i} by factor {scale_factor}")
            m = m.process("math.fft.resample", {"n": scale_factor})
            m.set_attr("apix_x", target_apix)
            m.set_attr("apix_y", target_apix)
            m.set_attr("apix_z", target_apix)
        models_rescaled.append(m)
    
    # -------------------------------------------------------------------------
    # Normalize and binarize models.
    # -------------------------------------------------------------------------
    normalized_models = []
    for i, m in enumerate(models_rescaled):
        if options.verbose > 1:
            print(f"[Verbose >1] Normalizing model {i}")
        nm = normalize_model(m, options)
        normalized_models.append(nm)
    
    if options.savesteps:
        for i, nm in enumerate(normalized_models):
            fname = os.path.join(options.output_dir, f"normalized_model_{i}.hdf")
            nm.write_image(fname, 0)
            if options.verbose > 0:
                print(f"[Verbose >0] Saved normalized model {i} to {fname}")
    
    # -------------------------------------------------------------------------
    # For each normalized model, compute its longest span and corresponding radius.
    # Build a species list with each species' normalized model, computed radius, and spherical mask.
    # -------------------------------------------------------------------------
    species_list = []
    for i, nm in enumerate(normalized_models):
        diag = calculate_structure_diagonal(nm, threshold=0.05)
        radius = diag / 2.0
        mask = get_sphere_mask(radius)
        source = nm.get_attr_default("source", f"model_{i}")
        species_list.append({"model": nm, "radius": radius, "mask": mask, "source": source})
        if options.verbose > 0:
            print(f"[Verbose >0] {source}: longest span = {diag:.2f} voxels, radius = {radius:.2f}")
    
    # Use the maximum radius among all species to report the "largest" particle.
    max_radius = max([s["radius"] for s in species_list])
    if options.verbose > 0:
        print(f"[Verbose >0] Largest particle radius among all species = {max_radius:.2f} voxels")
    
    # -------------------------------------------------------------------------
    # Determine total number of placements to aim for.
    # If --nptcls is not provided, use a default high number (e.g., 1000).
    # -------------------------------------------------------------------------
    total_n = options.nptcls if options.nptcls is not None else 1000
    
    # -------------------------------------------------------------------------
    # Pack particles using occupancy-based algorithm.
    # -------------------------------------------------------------------------
    placements = pack_particles(species_list, (options.tomonx, options.tomony, options.tomonz),
                                 total_n, options.equimolar, options.max_attempts, options.verbose)
    if options.verbose > 0:
        print(f"[Verbose >0] Total particles placed: {len(placements)}")
    
    if len(placements) == 0:
        sys.stderr.write("No particles could be placed. Exiting.\n")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Create tomogram and label volumes.
    # -------------------------------------------------------------------------
    tomo = EMData(options.tomonx, options.tomony, options.tomonz)
    if options.background > 0:
        tomo.to_one()
        tomo.mult(options.background)
    else:
        tomo.to_zero()
    label_tomo = tomo.copy()
    label_tomo.to_zero()
    
    # -------------------------------------------------------------------------
    # For each placement, retrieve the corresponding species model, apply random rotation 
    # and a small jitter (here, up to 1 voxel) and insert it into the tomogram.
    # -------------------------------------------------------------------------
    ptcl_count = 0
    for placement in placements:
        species_idx = placement["species_idx"]
        x, y, z = placement["x"], placement["y"], placement["z"]
        particle = species_list[species_idx]["model"].copy()
        # Create random rotation.
        t = Transform({
            "type": "eman",
            "az": random.uniform(0, 360),
            "alt": random.uniform(0, 360),
            "phi": random.uniform(0, 360)
        })
        particle.transform(t)
        # Apply small jitter (±1 voxel).
        jitter = 1
        x += random.randint(-jitter, jitter)
        y += random.randint(-jitter, jitter)
        z += random.randint(-jitter, jitter)
        tomo.insert_scaled_sum(particle, [x, y, z])
        bin_particle = particle.copy().process("threshold.binary", {"value": 0.5})
        label_tomo.insert_scaled_sum(bin_particle, [x, y, z])
        if options.verbose > 2:
            print(f"[Verbose >2] Inserted particle {ptcl_count} from {species_list[species_idx]['source']} at ({x},{y},{z}), rotation: {t.get_params('eman')}")
        ptcl_count += 1
    
    # -------------------------------------------------------------------------
    # Write final output volumes.
    # -------------------------------------------------------------------------
    output_tomo = os.path.join(options.output_dir, options.output)
    output_label = os.path.join(options.output_dir, options.output_label)
    tomo.write_image(output_tomo, 0)
    label_tomo.write_image(output_label, 0)
    if options.verbose > 0:
        print(f"[Verbose >0] Wrote tomogram to {output_tomo}")
        print(f"[Verbose >0] Wrote label map to {output_label}")
    
    # End EMAN2 logging.
    E2end(logger)
    elapsed = time.perf_counter() - start_time
    if options.verbose > 0:
        print(f"[Verbose >0] Elapsed time: {str(timedelta(seconds=elapsed))}")

##############################################################################
# Run main
##############################################################################
if __name__ == "__main__":
    main()
    sys.stdout.flush()


'''
#!/usr/bin/env python
"""
tomo_sim_pdb.py

Generates a simulated "ground truth" tomogram by placing copies of 3D models (loaded locally
or fetched from EMDB/PDB) at random positions and orientations in a larger volume.

Key features:
  - Grid spacing is determined from the longest span (diagonal of the bounding box) of each
    input structure. For total crowdedness (dilutionfactor=1) the spacing is equal to the
    longest span.
  - For each unique input (file, EMDB or PDB accession) the computed longest span is printed.
  - Verbose output is provided at several levels.
  - Output files are written in a numbered directory to avoid overwriting.
  - EMAN2 logger is initialized with E2init() and closed via E2end(logger).

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import math
import random
import argparse
import time
import tempfile
import shutil
import gzip
import numpy as np
from datetime import timedelta

# Import EMAN2 modules
from EMAN2 import EMData, EMUtil, Transform, EMArgumentParser, E2init, E2end
import requests

##############################################################################
# Helper Functions
##############################################################################
def make_output_dir(base, verbose=0):
    """Create a new numbered output directory (e.g. base_00, base_01, etc.)"""
    suffix = 0
    while True:
        dirname = f"{base}_{suffix:02d}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            if verbose > 0:
                print(f"[Verbose >0] Created output directory: {dirname}")
            return dirname
        suffix += 1

def calculate_structure_diagonal(model, threshold=0.05):
    """
    Calculate the diagonal (longest span) of the structure in 'model'.
    Finds the bounding box of voxels with density > threshold.
    Returns the Euclidean distance between min and max indices along all axes.
    If no voxel exceeds the threshold, returns model["nx"].
    """
    data = model.numpy()
    indices = np.nonzero(data > threshold)
    if indices[0].size == 0:
        return model["nx"]
    x_min, x_max = indices[0].min(), indices[0].max()
    y_min, y_max = indices[1].min(), indices[1].max()
    z_min, z_max = indices[2].min(), indices[2].max()
    diag = math.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)
    return diag

def mid_points(length, segment, step):
    """
    Return midpoints along a line of given 'length', for segments of size 'segment' stepping by 'step'.
    Uses <= to include the last block if it fits.
    """
    return [int(round(p + segment/2.0))
            for p in range(0, length, step)
            if (p + segment/2.0) <= (length - segment/2.0)]

def get_coords(options, spacing):
    """
    Generate grid coordinates for particle placement using the provided spacing.
    'spacing' is converted to an integer.
    """
    spacing = int(round(spacing))
    xs = mid_points(options.tomonx, spacing, spacing)
    ys = mid_points(options.tomony, spacing, spacing)
    zs = mid_points(options.tomonz, spacing, spacing)
    coords = []
    for x in xs:
        for y in ys:
            for z in zs:
                coords.append((x, y, z))
    return coords

def normalize_model(model, options):
    """
    Normalize the model:
      1) Shift so that the minimum becomes 0.
      2) Scale so that the range is [0,1].
      3) Binarize: set voxels to 1 if above (mean + threshold_sigma * std), else 0.
    Returns a new EMData object containing the binary data.
    """
    # Shift: subtract the minimum.
    min_val = model["minimum"]
    model.add(-min_val)
    # Scale: divide by the maximum.
    max_val = model["maximum"]
    if max_val != 0:
        model.div(max_val)
    # Binarize using the underlying NumPy array.
    data = model.numpy()
    mean_val = np.mean(data)
    std_val  = np.std(data)
    thresh   = mean_val + options.threshold_sigma * std_val
    bin_data = (data >= thresh).astype(np.float32)
    nx, ny, nz = bin_data.shape
    new_model = EMData(nx, ny, nz)
    new_model.set_data_string(bin_data.tobytes())
    new_model.set_attr_dict({"minimum": float(bin_data.min()),
                             "maximum": float(bin_data.max())})
    # Propagate the source attribute.
    new_model.set_attr("source", model.get_attr_default("source", "unknown"))
    return new_model

def fetch_emdb_map(acc, options):
    """
    Fetch an EMDB map (map.gz) for accession 'acc'. Adjust the URL/API as needed.
    """
    try:
        url = f"https://www.ebi.ac.uk/emdb/entry-map/{acc}.map.gz"
        if options.verbose > 1:
            print(f"[Verbose >1] Downloading EMDB map from {url}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            sys.stderr.write(f"Error downloading EMDB {acc}: HTTP {response.status_code}\n")
            return None
        tmp_dir = tempfile.mkdtemp()
        gz_path = os.path.join(tmp_dir, f"{acc}.map.gz")
        with open(gz_path, "wb") as f:
            f.write(response.content)
        try:
            vol = EMData(gz_path, 0)
        except Exception:
            mrc_path = os.path.join(tmp_dir, f"{acc}.mrc")
            with gzip.open(gz_path, "rb") as f_in, open(mrc_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            vol = EMData(mrc_path, 0)
        shutil.rmtree(tmp_dir)
        # Tag with source information.
        vol.set_attr("source", f"EMDB:{acc}")
        return vol
    except Exception as e:
        sys.stderr.write(f"Exception fetching EMDB {acc}: {e}\n")
        return None

def fetch_pdb_map(acc, options):
    """
    Fetch a PDB model (pdb1.gz) for accession 'acc' and convert it to a density map.
    This dummy version creates a volume of size box_size^3.
    Replace with an actual conversion (e.g., via e2pdb2mrc.py) as needed.
    """
    try:
        url = f"https://files.rcsb.org/download/{acc}.pdb1.gz"
        if options.verbose > 1:
            print(f"[Verbose >1] Downloading PDB model from {url}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            sys.stderr.write(f"Error downloading PDB {acc}: HTTP {response.status_code}\n")
            return None
        tmp_dir = tempfile.mkdtemp()
        gz_path = os.path.join(tmp_dir, f"{acc}.pdb.gz")
        with open(gz_path, "wb") as f:
            f.write(response.content)
        pdb_path = os.path.join(tmp_dir, f"{acc}.pdb")
        with gzip.open(gz_path, "rb") as f_in, open(pdb_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        # Dummy conversion: create a volume of size box_size^3.
        vol = EMData(options.box_size, options.box_size, options.box_size)
        vol.to_one()
        shutil.rmtree(tmp_dir)
        # Tag with source information.
        vol.set_attr("source", f"PDB:{acc}")
        return vol
    except Exception as e:
        sys.stderr.write(f"Exception fetching PDB {acc}: {e}\n")
        return None

##############################################################################
# Main function
##############################################################################
def main():
    start_time = time.perf_counter()
    
    # -------------------------------------------------------------------------
    # Parse Arguments
    # -------------------------------------------------------------------------
    parser = EMArgumentParser(usage="tomo_sim_pdb.py [options]", version="1.0")
    parser.add_argument("--apix", type=float, default=1.0,
                        help="Target pixel size for simulated volume. Models will be rescaled if needed.")
    parser.add_argument("--tomonx", type=int, default=256, help="Tomogram size in x.")
    parser.add_argument("--tomony", type=int, default=256, help="Tomogram size in y.")
    parser.add_argument("--tomonz", type=int, default=168, help="Tomogram size in z.")
    parser.add_argument("--box_size", type=int, default=64,
                        help="Size (in voxels) for each particle's bounding box.")
    parser.add_argument("--dilutionfactor", type=int, default=1,
                        help="Dilution factor; 1 means maximum crowdedness, 2 means half as many particles, etc.")
    parser.add_argument("--nptcls", type=int, default=None,
                        help="Total number of particles to insert (overrides dilution if lower).")
    parser.add_argument("--input", type=str, default=None,
                        help="Comma-separated list of local volumes (MRC/HDF).")
    parser.add_argument("--input_emdb", type=str, default=None,
                        help="Comma-separated list of EMDB accession numbers.")
    parser.add_argument("--input_pdb", type=str, default=None,
                        help="Comma-separated list of PDB accession numbers.")
    parser.add_argument("--threshold_sigma", type=float, default=3.0,
                        help="Sigma threshold for binarizing (default=3.0).")
    parser.add_argument("--savesteps", action="store_true", default=False,
                        help="Save intermediate normalized models to disk.")
    parser.add_argument("--background", type=float, default=0.0,
                        help="Background density (e.g., 0.71428571428 for ice; 0 for black).")
    parser.add_argument("--output", type=str, default="simulated_tomogram.hdf",
                        help="Filename for the final tomogram.")
    parser.add_argument("--output_label", type=str, default="simulated_labels.hdf",
                        help="Filename for the label (segmentation) map.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory; if not given, a numbered directory (e.g., tomo_gt_pdb_00) is created.")
    parser.add_argument("--verbose", "-v", type=int, default=0,
                        help="Verbosity level [0..9]. Higher means more output.")
    
    (options, args) = parser.parse_args()
    
    # Initialize EMAN2 logging (pass logger to E2end)
    logger = E2init(sys.argv, options.ppid if hasattr(options, "ppid") else -1)
    
    if options.verbose > 0:
        print(f"[Verbose >0] Starting simulation with apix={options.apix}, box_size={options.box_size}, "
              f"volume=({options.tomonx}, {options.tomony}, {options.tomonz})")
    
    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------
    if options.output_dir is None:
        options.output_dir = make_output_dir("tomo_gt_pdb", verbose=options.verbose)
    else:
        if not os.path.exists(options.output_dir):
            os.makedirs(options.output_dir)
            if options.verbose > 0:
                print(f"[Verbose >0] Created output directory: {options.output_dir}")
    
    # -------------------------------------------------------------------------
    # Load input models (local, EMDB, PDB) and tag with their source.
    # -------------------------------------------------------------------------
    models = []
    if options.input:
        file_list = [f.strip() for f in options.input.split(',') if f.strip()]
        for f in file_list:
            if options.verbose > 0:
                print(f"[Verbose >0] Loading local volume: {f}")
            try:
                n_images = EMUtil.get_image_count(f)
                if n_images > 1:
                    for i in range(n_images):
                        vol = EMData(f, i)
                        vol.set_attr("source", f)
                        models.append(vol)
                        if options.verbose > 1:
                            print(f"[Verbose >1]  Loaded sub-volume {i} from {f}")
                else:
                    vol = EMData(f, 0)
                    vol.set_attr("source", f)
                    models.append(vol)
            except Exception as e:
                sys.stderr.write(f"Error loading file {f}: {e}\n")
    if options.input_emdb:
        emdb_list = [s.strip() for s in options.input_emdb.split(',') if s.strip()]
        for acc in emdb_list:
            if options.verbose > 0:
                print(f"[Verbose >0] Fetching EMDB map: {acc}")
            vol = fetch_emdb_map(acc, options)
            if vol:
                models.append(vol)
    if options.input_pdb:
        pdb_list = [s.strip() for s in options.input_pdb.split(',') if s.strip()]
        for acc in pdb_list:
            if options.verbose > 0:
                print(f"[Verbose >0] Fetching PDB model: {acc}")
            vol = fetch_pdb_map(acc, options)
            if vol:
                models.append(vol)
    
    if len(models) == 0:
        sys.stderr.write("No models were loaded. Exiting.\n")
        sys.exit(1)
    if options.verbose > 0:
        print(f"[Verbose >0] Loaded {len(models)} model(s).")
    
    # -------------------------------------------------------------------------
    # Determine target apix and rescale models if needed
    # -------------------------------------------------------------------------
    target_apix = options.apix
    for m in models:
        apix_val = m.get_attr_default("apix_x", options.apix)
        if apix_val > target_apix:
            target_apix = apix_val
    if options.verbose > 0:
        print(f"[Verbose >0] Final target apix = {target_apix}")
    
    models_rescaled = []
    for i, m in enumerate(models):
        apix_val = m.get_attr_default("apix_x", options.apix)
        if abs(apix_val - target_apix) > 1e-5:
            scale_factor = apix_val / target_apix
            if options.verbose > 1:
                print(f"[Verbose >1] Rescaling model {i} by factor {scale_factor}")
            m = m.process("math.fft.resample", {"n": scale_factor})
            m.set_attr("apix_x", target_apix)
            m.set_attr("apix_y", target_apix)
            m.set_attr("apix_z", target_apix)
        models_rescaled.append(m)
    
    # -------------------------------------------------------------------------
    # Normalize and binarize models
    # -------------------------------------------------------------------------
    normalized_models = []
    for i, m in enumerate(models_rescaled):
        if options.verbose > 1:
            print(f"[Verbose >1] Normalizing model {i}")
        nm = normalize_model(m, options)
        normalized_models.append(nm)
    
    if options.savesteps:
        for i, nm in enumerate(normalized_models):
            fname = os.path.join(options.output_dir, f"normalized_model_{i}.hdf")
            nm.write_image(fname, 0)
            if options.verbose > 0:
                print(f"[Verbose >0] Saved normalized model {i} to {fname}")
    
    # -------------------------------------------------------------------------
    # Compute longest span for each unique input source and print to terminal.
    # -------------------------------------------------------------------------
    span_dict = {}
    for nm in normalized_models:
        source = nm.get_attr_default("source", "unknown")
        diag = calculate_structure_diagonal(nm, threshold=0.05)
        if source not in span_dict or diag > span_dict[source]:
            span_dict[source] = diag
    for source, diag in span_dict.items():
        if options.verbose > 0:
            print(f"[Verbose >0] Longest span for {source}: {diag:.2f} voxels")
    
    # Use the maximum span among all unique inputs as grid spacing.
    grid_spacing = max(span_dict.values()) if span_dict else options.box_size
    if options.verbose > 0:
        print(f"[Verbose >0] Using grid spacing = {grid_spacing:.2f} voxels")
    
    # -------------------------------------------------------------------------
    # Generate grid coordinates using the computed spacing, apply dilution.
    # -------------------------------------------------------------------------
    coords = get_coords(options, spacing=grid_spacing)
    random.shuffle(coords)
    ncoords = len(coords)
    if ncoords == 0:
        sys.stderr.write("No grid coordinates generated. Check volume size and grid spacing.\n")
        sys.exit(1)
    nptclsmax = ncoords // options.dilutionfactor
    if options.nptcls is None:
        nptcls = nptclsmax
    else:
        nptcls = min(options.nptcls, nptclsmax)
    coords = coords[:nptcls]
    if options.verbose > 0:
        print(f"[Verbose >0] Total grid positions: {ncoords}, after dilution: {nptclsmax}, final particles: {nptcls}")
    
    # -------------------------------------------------------------------------
    # Insert particles into the tomogram.
    # -------------------------------------------------------------------------
    num_models = len(normalized_models)
    n_per_model = int(math.ceil(nptcls / float(num_models)))
    if options.verbose > 0:
        print(f"[Verbose >0] Inserting {nptcls} particles (~{n_per_model} per model)")
    
    tomo = EMData(options.tomonx, options.tomony, options.tomonz)
    if options.background > 0:
        tomo.to_one()
        tomo.mult(options.background)
    else:
        tomo.to_zero()
    
    label_tomo = tomo.copy()
    label_tomo.to_zero()
    
    ptcl_count = 0
    for i, model in enumerate(normalized_models):
        if options.verbose > 1:
            print(f"[Verbose >1] Inserting copies of model {i}")
        for j in range(n_per_model):
            if ptcl_count >= nptcls:
                break
            particle = model.copy()
            # Create random rotation using a dictionary in the Transform constructor.
            t = Transform({
                "type": "eman",
                "az": random.uniform(0, 360),
                "alt": random.uniform(0, 360),
                "phi": random.uniform(0, 360)
            })
            particle.transform(t)
            x, y, z = coords[ptcl_count]
            jitter = options.box_size // 4
            x += random.randint(-jitter, jitter)
            y += random.randint(-jitter, jitter)
            z += random.randint(-jitter, jitter)
            tomo.insert_scaled_sum(particle, [x, y, z])
            bin_particle = particle.copy().process("threshold.binary", {"value": 0.5})
            label_tomo.insert_scaled_sum(bin_particle, [x, y, z])
            if options.verbose > 2:
                print(f"[Verbose >2] Inserted particle {ptcl_count} at ({x},{y},{z}), rotation: {t.get_params('eman')}")
            ptcl_count += 1
    
    # -------------------------------------------------------------------------
    # Write final output volumes
    # -------------------------------------------------------------------------
    output_tomo = os.path.join(options.output_dir, options.output)
    output_label = os.path.join(options.output_dir, options.output_label)
    tomo.write_image(output_tomo, 0)
    label_tomo.write_image(output_label, 0)
    if options.verbose > 0:
        print(f"[Verbose >0] Wrote tomogram to {output_tomo}")
        print(f"[Verbose >0] Wrote label map to {output_label}")
    
    # End EMAN2 logging (pass the logger)
    E2end(logger)
    elapsed = time.perf_counter() - start_time
    if options.verbose > 0:
        print(f"[Verbose >0] Elapsed time: {str(timedelta(seconds=elapsed))}")

##############################################################################
# Run main
##############################################################################
if __name__ == "__main__":
    main()
    sys.stdout.flush()


'''
