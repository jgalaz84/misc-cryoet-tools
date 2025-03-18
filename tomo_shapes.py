#!/usr/bin/env python
"""
tomo_sim_shapes.py

Author: Jesus Galaz-Montoya
Date: 04/2020
Last modification: 03/2025

Generates a simulated "ground truth" tomogram by placing copies of 3D geometric shapes
inside a larger volume. Two input modes are available:
  1) --input_shapes: Supply a comma-separated list of shape names (or "all") for on-the-fly generation.
     Valid names: cube, sphere, sheet, prism, pyramid, cylinder, disc, ellipsoid, cross2, cross3.
  2) --input: Supply pre-made shape files (MRC/HDF).

Each shape type is treated as a distinct "species." A generic scaling step ensures that every
shape is uniformly scaled so that its maximum span (diagonal) does not exceed the side length of
the base volume. This allows free rotation without clipping. A KD-Tree–based packing algorithm
quickly finds free positions. In equimolar mode (--equimolar), species are placed round-robin;
otherwise, they are placed in descending order of radius.

No thresholding is applied during insertion. If --varydensity is off, label volumes are binarized
(after placement, any voxel > 0 becomes 1). If a placement attempt for a species takes more than 60
seconds, that species is removed from further placement.

Outputs:
  - One combined tomogram (e.g., simulated_tomogram.mrc)
  - One label volume per species (e.g., label_cube.mrc, label_sphere.mrc, etc.)

Example:
  python tomo_sim_shapes.py --input_shapes cube,sphere,prism --tomonx 256 --tomony 256 --tomonz 168 \
      --dilutionfactor 1 --equimolar --varydensity --varysize --savesteps --background 0.71428571428 --verbose 10
"""

import os, sys, math, random, time
import numpy as np
from datetime import timedelta

# EMAN2 modules
from EMAN2 import EMData, EMUtil, Transform, EMArgumentParser, E2init, E2end
from EMAN2_utils import clip3d  # ensure clip3d is available

# For KD-Tree packing
try:
    from scipy.spatial import KDTree
except ImportError:
    sys.stderr.write("Error: SciPy is required for KDTree-based packing. Please install scipy.\n")
    sys.exit(1)

##############################################################################
# Generic scaling: Ensure shape's diagonal <= box_size
##############################################################################
def scale_to_box(shape, box_size, threshold=0.05):
    """
    Scale the input shape uniformly so that its maximum span (diagonal)
    does not exceed the box_size.
    """
    current_span = calculate_structure_diagonal(shape, threshold)
    if current_span > box_size:
        factor = box_size / current_span
        shape.process_inplace("xform.scale", {"scale": factor, "clip": box_size})
    return shape

##############################################################################
# Utility: Create output directory
##############################################################################
def make_output_dir(base, verbose=0):
    suffix = 0
    while True:
        dirname = f"{base}_{suffix:02d}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            if verbose:
                print(f"[Verbose] Created output directory: {dirname}")
            return dirname
        suffix += 1

##############################################################################
# Compute shape diagonal and (optional) spherical mask
##############################################################################
def calculate_structure_diagonal(model, threshold=0.05):
    data = model.numpy()
    idx = np.nonzero(data > threshold)
    if idx[0].size == 0:
        return model["nx"]
    x_min, x_max = idx[0].min(), idx[0].max()
    y_min, y_max = idx[1].min(), idx[1].max()
    z_min, z_max = idx[2].min(), idx[2].max()
    diag = math.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)
    return diag

def get_sphere_mask(radius):
    r_int = int(math.ceil(radius))
    offsets = []
    for dx in range(-r_int, r_int+1):
        for dy in range(-r_int, r_int+1):
            for dz in range(-r_int, r_int+1):
                if dx*dx + dy*dy + dz*dz <= radius*radius:
                    offsets.append((dx,dy,dz))
    return offsets

##############################################################################
# Normalization: shift min->0, scale max->1. (No extra thresholding by default.)
##############################################################################
def normalize_model(model, options):
    data = model.numpy()
    # If uniform, return model as is
    if data.min() == data.max():
        return model.copy()
    data = data - data.min()
    max_val = data.max()
    if max_val != 0:
        data /= max_val
    if options.threshold_sigma > 0.0:
        thr = data.mean() + options.threshold_sigma * data.std()
        bin_data = (data >= thr).astype(np.float32)
    else:
        bin_data = data.astype(np.float32)
    nx, ny, nz = bin_data.shape
    new_model = EMData(nx, ny, nz)
    new_model.set_data_string(bin_data.tobytes())
    new_model.set_attr_dict({
        "minimum": float(bin_data.min()),
        "maximum": float(bin_data.max())
    })
    new_model.set_attr("source", model.get_attr_default("source", "unknown"))
    return new_model

##############################################################################
# KD-Tree-based packing function with time-limit (60s per species placement)
##############################################################################
def pack_particles_kdtree(species_list, volume_dims, total_n, equimolar, max_attempts, verbose):
    """
    Place up to total_n shapes using a KD-Tree for fast collision checking.
    For each species, if a placement attempt takes >60s, that species is removed from further attempts.
    Returns a list of placements, each a dict with keys: "x", "y", "z", "species_idx".
    """
    X, Y, Z = volume_dims
    placements = []
    all_centers = []  # list of (x,y,z)
    all_radii = []    # parallel list of radii
    kd = None

    def rebuild_kdtree():
        nonlocal kd
        if len(all_centers)==0:
            kd = None
        else:
            kd = KDTree(all_centers)

    def can_place_center(cx, cy, cz, cr):
        if kd is None:
            return True
        # Query with radius = cr + max(existing radii) + margin
        if len(all_radii)==0:
            return True
        maxr = max(all_radii)
        query_r = cr + maxr + 2
        neighbors = kd.query_ball_point([cx,cy,cz], query_r)
        for ni in neighbors:
            nx, ny, nz = all_centers[ni]
            nr = all_radii[ni]
            if math.sqrt((cx - nx)**2 + (cy - ny)**2 + (cz - nz)**2) < (cr + nr):
                return False
        return True

    def attempt_place_one(sp_idx):
        """Try to place one shape of species sp_idx. Return (x,y,z) or None if timed out."""
        r = species_list[sp_idx]["radius"]
        start_time = time.time()
        for attempt in range(max_attempts):
            if time.time() - start_time > 60:
                return None
            # Generate a random coordinate that allows margin for radius r
            cx = random.uniform(r, X - r)
            cy = random.uniform(r, Y - r)
            cz = random.uniform(r, Z - r)
            if can_place_center(cx, cy, cz, r):
                return (cx, cy, cz)
        return None

    placed_count = 0
    if equimolar:
        active = list(range(len(species_list)))
        round_num = 0
        while placed_count < total_n and active:
            round_num += 1
            if verbose > 2:
                print(f"[Verbose >2] KD-Tree Equimolar Round {round_num}, active species: {active}")
            placed_this_round = False
            for sp_idx in active[:]:
                coord = attempt_place_one(sp_idx)
                if coord is None:
                    if verbose:
                        print(f"[Verbose] Species '{species_list[sp_idx]['source']}' timed out in round {round_num}, removing.")
                    active.remove(sp_idx)
                else:
                    (cx, cy, cz) = coord
                    all_centers.append((cx, cy, cz))
                    all_radii.append(species_list[sp_idx]["radius"])
                    rebuild_kdtree()
                    placements.append({"x": cx, "y": cy, "z": cz, "species_idx": sp_idx})
                    placed_count += 1
                    placed_this_round = True
                    if verbose > 2:
                        print(f"[Verbose >2] Round {round_num}: Placed {species_list[sp_idx]['source']} at ({int(cx)},{int(cy)},{int(cz)})")
                    if placed_count >= total_n:
                        break
            if not placed_this_round:
                if verbose:
                    print("[Verbose] No placements in this round; volume may be full or species timed out.")
                print("No more shapes fit or species timed out; stopping equimolar packing.")
                break
    else:
        sorted_idxs = sorted(range(len(species_list)),
                              key=lambda i: species_list[i]["radius"],
                              reverse=True)
        for sp_idx in sorted_idxs:
            while placed_count < total_n:
                coord = attempt_place_one(sp_idx)
                if coord is None:
                    if verbose:
                        print(f"[Verbose] Stopping placements for species '{species_list[sp_idx]['source']}' (time/attempts).")
                    break
                (cx, cy, cz) = coord
                all_centers.append((cx, cy, cz))
                all_radii.append(species_list[sp_idx]["radius"])
                rebuild_kdtree()
                placements.append({"x": cx, "y": cy, "z": cz, "species_idx": sp_idx})
                placed_count += 1
                if verbose > 2:
                    print(f"[Verbose >2] Placed {species_list[sp_idx]['source']} at ({int(cx)},{int(cy)},{int(cz)})")
                if placed_count >= total_n:
                    break
    return placements

##############################################################################
# -------------- Shape generation functions ----------------------------------
##############################################################################
def cube(options, vol_in):
    """
    Generate an inscribed cube. Instead of filling the whole volume, we create a cube
    that is centered and has side length = box_size / √3, so its diagonal equals box_size.
    """
    L = vol_in["nx"]
    new_side = L / math.sqrt(3)
    margin = (L - new_side) / 2.0
    return vol_in.process("mask.zeroedge3d", {
        "x0": margin, "x1": margin,
        "y0": margin, "y1": margin,
        "z0": margin, "z1": margin
    })

def sphere(options, vol_in):
    """Generate a sphere that exactly fits the box: diameter equals box_size."""
    L = vol_in["nx"]
    return vol_in.process("mask.sharp", {"outer_radius": L/2.0})

def sheet(options, vol_in):
    """Generate a thin sheet in the center."""
    L = vol_in["nx"]
    return vol_in.process("mask.zeroedge3d", {
        "x0": 1, "x1": 1,
        "y0": 1, "y1": 1,
        "z0": int(L/2-1), "z1": int(L/2-1)
    })

def prism(options, vol_in, thickness=None):
    """
    Generate a rectangular prism that is inscribed in the box.
    For instance, we choose dimensions (w, h, w) such that the diagonal equals box_size.
    One simple choice is: w = L/1.5 and h = L/3.
    """
    L = vol_in["nx"]
    w = L / 1.5
    h = L / 3.0
    margin_x = (L - w) / 2.0
    margin_y = (L - h) / 2.0
    margin_z = (L - w) / 2.0
    return vol_in.process("mask.zeroedge3d", {
        "x0": margin_x, "x1": margin_x,
        "y0": margin_y, "y1": margin_y,
        "z0": margin_z, "z1": margin_z
    })

def pyramid(options, vol_in):
    """
    Generate a pyramid inscribed in the box.
    Here we use a simple approach: leave a margin equal to one-third of the box on each side.
    """
    L = vol_in["nx"]
    margin = L / 3.0
    return vol_in.process("mask.zeroedge3d", {
        "x0": margin, "x1": margin,
        "y0": margin, "y1": margin,
        "z0": margin, "z1": margin
    })

def cylinder(options, vol_in):
    """Generate a cylinder that fits in the box: use radius = L/4."""
    L = vol_in["nx"]
    return vol_in.process("testimage.cylinder", {"height": L, "radius": L/4.0})

def disc(options, vol_in):
    """Generate a disc that fits in the box."""
    L = vol_in["nx"]
    return vol_in.process("testimage.disc", {"major": L/4.0, "minor": L/6.0, "height": L/6.0})

def ellipsoid(options, vol_in):
    """Generate an ellipsoid inscribed in the box."""
    L = vol_in["nx"]
    vol_in.to_zero()
    return vol_in.process("testimage.ellipsoid", {"a": L/3.0, "b": L/4.0, "c": L/5.0, "fill": 1})

def cross2(options, vol_in, thickness=None):
    """Generate a cross2 shape."""
    L = vol_in["nx"]
    e1 = prism(options, vol_in, thickness)
    e2 = e1.copy()
    t = Transform({"type": "eman", "az": 0, "alt": 90, "phi": 0})
    e2.transform(t)
    return e1 + e2

def cross3(options, vol_in, thickness=None):
    """Generate a cross3 shape."""
    L = vol_in["nx"]
    e1 = prism(options, vol_in, thickness)
    e2 = e1.copy()
    t = Transform({"type": "eman", "az": 0, "alt": 90, "phi": 0})
    e2.transform(t)
    e3 = e1.copy()
    t = Transform({"type": "eman", "az": 0, "alt": 90, "phi": 90})
    e3.transform(t)
    return e1 + e2 + e3

def gen_layers(options, shape):
    """Add concentric layers (shells) to the shape if requested."""
    gr = (1+5**0.5)/2
    layered = shape.copy()
    L = shape["nx"]
    for i in range(options.layers):
        s2 = shape.copy()
        s2_shrunk = s2.process("math.fft.resample", {"n": gr**(i+1)})
        if i % 2 == 0:
            s2_shrunk *= -1
        layered += clip3d(s2_shrunk, L)
    if options.background:
        layered *= (1.0 / options.background)
    return layered

##############################################################################
# Main function
##############################################################################
def main():
    start_time = time.perf_counter()  # Start timer

    parser = EMArgumentParser(usage="tomo_sim_shapes.py [options]", version="1.0")
    parser.add_argument("--apix", type=float, default=1.0, help="Pixel size.")
    parser.add_argument("--tomonx", type=int, default=256, help="Tomogram size in x.")
    parser.add_argument("--tomony", type=int, default=256, help="Tomogram size in y.")
    parser.add_argument("--tomonz", type=int, default=168, help="Tomogram size in z.")
    parser.add_argument("--box_size", type=int, default=64, help="Base volume size for each shape.")
    parser.add_argument("--dilutionfactor", type=int, default=1, help="1=full crowding, 2=half, etc.")
    parser.add_argument("--nptcls", type=int, default=None, help="Total shapes to place. Default 1000 if not set.")
    parser.add_argument("--input_shapes", type=str, default="all", help="Comma-separated shape names or 'all'.")
    parser.add_argument("--input", type=str, default=None, help="Comma-separated pre-made shape files (MRC/HDF).")
    parser.add_argument("--ndiversity", type=int, default=1, help="(Ignored if --input_shapes is provided explicitly.)")
    parser.add_argument("--varydensity", action="store_true", default=False,
                        help="Random ±10% density variation. If off, label volumes are binarized at the end.")
    parser.add_argument("--varysize", action="store_true", default=False, help="Random scale factor between 0.5 and 1.0.")
    parser.add_argument("--softedges", action="store_true", default=False, help="(Optional) Apply soft edges (blur).")
    parser.add_argument("--layers", type=int, default=None, help="Add up to 4 concentric shells if set.")
    parser.add_argument("--half_shapes", type=str, default=None, help="'all' or 'some' to clip shapes in half.")
    parser.add_argument("--zc", type=int, default=None, help="Force z coordinate for all shapes.")
    parser.add_argument("--equimolar", action="store_true", default=False, help="Round-robin distribution among species.")
    parser.add_argument("--max_attempts", type=int, default=50000, help="Max random attempts per shape placement.")
    parser.add_argument("--background", type=float, default=0.0, help="Background density (e.g., 0.71428571428 for ice).")
    parser.add_argument("--threshold_sigma", type=float, default=0.0, help="Extra thresholding; default 0.0 disables.")
    parser.add_argument("--output", type=str, default="simulated_tomogram.hdf", help="Output tomogram filename (default HDF).")
    parser.add_argument("--output_label", type=str, default="simulated_labels.hdf", help="(Unused; separate labels are generated per species).")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory; if not provided, a numbered directory is created.")
    parser.add_argument("--hdfoutput", action="store_true", default=False, help="Use HDF output; otherwise, use MRC.")
    parser.add_argument("--savesteps", action="store_true", default=False, help="Save intermediate shape volumes.")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbosity level [0-9].")
    (options, args) = parser.parse_args()

    if not options.hdfoutput:
        if not options.output.lower().endswith(".mrc"):
            options.output = os.path.splitext(options.output)[0] + ".mrc"

    logger = E2init(sys.argv, options.ppid if hasattr(options, "ppid") else -1)
    if options.verbose:
        print(f"[Verbose] Starting simulation: box_size={options.box_size}, volume=({options.tomonx}, {options.tomony}, {options.tomonz})")
    if options.output_dir is None:
        options.output_dir = make_output_dir("tomo_gt_shapes", verbose=options.verbose)
    else:
        if not os.path.exists(options.output_dir):
            os.makedirs(options.output_dir)
            if options.verbose:
                print(f"[Verbose] Created output directory: {options.output_dir}")

    ##############################################################################
    # Build species list: from pre-made shape files and from --input_shapes.
    ##############################################################################
    species_list = []
    # 1) Pre-made shape files
    if options.input:
        file_list = [f.strip() for f in options.input.split(',') if f.strip()]
        for f in file_list:
            try:
                vol = EMData(f, 0)
                vol.set_attr("source", os.path.basename(f))
                norm_vol = normalize_model(vol, options)
                norm_vol = scale_to_box(norm_vol, options.box_size, threshold=0.05)
                diag = calculate_structure_diagonal(norm_vol, threshold=0.05)
                radius = diag / 2.0
                species_list.append({"model": norm_vol, "radius": radius, "source": os.path.basename(f)})
                if options.verbose:
                    print(f"[Verbose] Loaded shape file {f}: diag={diag:.2f}, radius={radius:.2f}")
            except Exception as e:
                sys.stderr.write(f"Error loading shape file {f}: {e}\n")
    # 2) Generated shapes from --input_shapes (ndiversity is ignored if provided)
    shape_dict = {
        "cube": cube,
        "sphere": sphere,
        "sheet": sheet,
        "prism": prism,
        "pyramid": pyramid,
        "cylinder": cylinder,
        "disc": disc,
        "ellipsoid": ellipsoid,
        "cross2": cross2,
        "cross3": cross3
    }
    if options.input_shapes:
        if options.input_shapes.lower() == "all":
            shape_names = list(shape_dict.keys())
        else:
            shape_names = [n.strip() for n in options.input_shapes.split(',') if n.strip()]
        print(f"\nParsed shape_names: {shape_names}")
        for name in shape_names:
            if name not in shape_dict:
                sys.stderr.write(f"Shape {name} not recognized; skipping.\n")
                continue
            try:
                base_vol = EMData(options.box_size, options.box_size, options.box_size)
                base_vol.to_one()
                shape_em = shape_dict[name](options, base_vol.copy())
                if options.layers and options.layers > 0:
                    shape_em = gen_layers(options, shape_em)
                shape_em.set_attr("source", name)
                norm_shape = normalize_model(shape_em, options)
                norm_shape = scale_to_box(norm_shape, options.box_size, threshold=0.05)
                diag = calculate_structure_diagonal(norm_shape, threshold=0.05)
                radius = diag / 2.0
                species_list.append({"model": norm_shape, "radius": radius, "source": name})
                if options.verbose:
                    print(f"[Verbose] Generated shape {name}: diag={diag:.2f}, radius={radius:.2f}")
                if options.savesteps:
                    outname = os.path.join(options.output_dir, f"{name}_base.hdf")
                    norm_shape.write_image(outname, 0)
                    if options.verbose > 1:
                        print(f"[Verbose] Saved shape {name} -> {outname}")
            except Exception as e:
                sys.stderr.write(f"Error generating shape {name}: {e}\n")
    if len(species_list) == 0:
        sys.stderr.write("No shapes available. Exiting.\n")
        sys.exit(1)

    ##############################################################################
    # Determine total placements after dilution.
    ##############################################################################
    final_n = options.nptcls if options.nptcls is not None else 1000
    final_n = int(math.floor(final_n / float(options.dilutionfactor)))
    if options.verbose:
        print(f"[Verbose] final total_n after dilution = {final_n}")

    ##############################################################################
    # Pack shapes using KD-Tree-based approach.
    ##############################################################################
    placements = pack_particles_kdtree(species_list, (options.tomonx, options.tomony, options.tomonz),
                                       final_n, options.equimolar, options.max_attempts, options.verbose)
    if options.verbose:
        print(f"[Verbose] Total shapes placed: {len(placements)}")
    if len(placements) == 0:
        sys.stderr.write("No particles placed. Exiting.\n")
        E2end(logger)
        sys.exit(1)

    ##############################################################################
    # Create final tomogram.
    ##############################################################################
    tomo = EMData(options.tomonx, options.tomony, options.tomonz)
    if options.background > 0:
        tomo.to_one()
        tomo.mult(options.background)
    else:
        tomo.to_zero()

    ##############################################################################
    # Create separate label volumes (one per species).
    ##############################################################################
    label_volumes = {}
    for sp in species_list:
        src = sp["source"]
        lbl = EMData(options.tomonx, options.tomony, options.tomonz)
        lbl.to_zero()
        label_volumes[src] = lbl

    ##############################################################################
    # Insert placed shapes into tomogram and corresponding label volumes.
    ##############################################################################
    ptcl_count = 0
    for p in placements:
        i = p["species_idx"]
        x, y, z = p["x"], p["y"], p["z"]
        sp = species_list[i]
        src = sp["source"]
        shape_c = sp["model"].copy()

        # Apply density variation if enabled.
        if options.varydensity:
            factor = random.uniform(0.9, 1.1)
            shape_c.mult(factor)

        # Apply size variation if enabled.
        effective_scale = 1.0
        if options.varysize:
            effective_scale = random.uniform(0.5, 1.0)
            shape_c.process_inplace("xform.scale", {"scale": effective_scale, "clip": options.box_size})
            if options.verbose > 2:
                effective_radius = sp["radius"] * effective_scale
                print(f"[Verbose >2] Placing {src}: base radius {sp['radius']:.2f}, scale {effective_scale:.2f}, effective diameter {2*effective_radius:.2f}")

        # Optionally clip half of the shape.
        if options.half_shapes:
            do_clip = False
            if options.half_shapes.lower() == "all":
                do_clip = True
            elif options.half_shapes.lower() == "some":
                do_clip = (ptcl_count % 2 == 0)
            if do_clip:
                axis = random.choice(["x", "y", "z"])
                direction = random.choice([-1, 1])
                if axis == "x":
                    if direction < 0:
                        shape_c.process_inplace("mask.zeroedge3d", {"x0": options.box_size/2, "x1": 0})
                    else:
                        shape_c.process_inplace("mask.zeroedge3d", {"x0": 0, "x1": options.box_size/2})
                elif axis == "y":
                    if direction < 0:
                        shape_c.process_inplace("mask.zeroedge3d", {"y0": options.box_size/2, "y1": 0})
                    else:
                        shape_c.process_inplace("mask.zeroedge3d", {"y0": 0, "y1": options.box_size/2})
                elif axis == "z":
                    if direction < 0:
                        shape_c.process_inplace("mask.zeroedge3d", {"z0": options.box_size/2, "z1": 0})
                    else:
                        shape_c.process_inplace("mask.zeroedge3d", {"z0": 0, "z1": options.box_size/2})
        # Apply a random rotation.
        t = Transform({"type": "eman",
                       "az": random.uniform(0, 360),
                       "alt": random.uniform(0, 360),
                       "phi": random.uniform(0, 360)})
        shape_c.transform(t)

        # Add small jitter.
        jitter = 1
        x += random.randint(-jitter, jitter)
        y += random.randint(-jitter, jitter)
        z += random.randint(-jitter, jitter)
        if options.zc is not None:
            z = options.zc

        tomo.insert_scaled_sum(shape_c, [x, y, z])
        label_volumes[src].insert_scaled_sum(shape_c, [x, y, z])
        ptcl_count += 1

    ##############################################################################
    # If varydensity is off, binarize label volumes (any voxel > 0 becomes 1.0)
    ##############################################################################
    if not options.varydensity:
        if options.verbose:
            print("[Verbose] Binarizing label volumes since varydensity is off.")
        for sp in species_list:
            src = sp["source"]
            lbl = label_volumes[src]
            lbl.process_inplace("threshold.belowtozero", {"minval": 0.0})
            lbl.process_inplace("threshold.binarize", {"value": 0.0})

    ##############################################################################
    # Write final outputs.
    ##############################################################################
    out_tomo = os.path.join(options.output_dir, options.output)
    tomo.write_image(out_tomo, 0)
    if options.verbose:
        print(f"[Verbose] Wrote tomogram -> {out_tomo}")

    for sp in species_list:
        src = sp["source"]
        lbl = label_volumes[src]
        # Check if label volume appears empty
        if lbl["maximum"] == 0.0 and lbl["sigma"] == 0.0 and lbl["minimum"] == 0.0:
            print(f"WARNING: label volume for species '{src}' is empty (sigma, max, and min are all 0).")
        lblname = f"label_{src}"
        if not options.hdfoutput and not lblname.lower().endswith(".mrc"):
            lblname += ".mrc"
        elif options.hdfoutput and not lblname.lower().endswith(".hdf"):
            lblname += ".hdf"
        out_lbl = os.path.join(options.output_dir, lblname)
        lbl.write_image(out_lbl, 0)
        if options.verbose:
            print(f"[Verbose] Wrote label map for '{src}' -> {out_lbl}")

    E2end(logger)
    elapsed = time.perf_counter() - start_time
    if options.verbose:
        print(f"[Verbose] Elapsed time: {str(timedelta(seconds=elapsed))}")

if __name__ == "__main__":
    main()
    sys.stdout.flush()



'''
#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 04/2020; last modification: 09/2024


from past.utils import old_div
from builtins import range

import math
import numpy as np
import random
import os
import sys

from EMAN2 import *
from EMAN2_utils import *
from EMAN2jsondb import JSTask,jsonclasses

import time
from datetime import timedelta


def main():
	start = time.perf_counter()

	progname = os.path.basename(sys.argv[0])
	usage = """prog [options]
	This programs can generate 3D shapes and multiple copies of 3D shapes (optionally layered and softened) at different scales and in different orientations, 
	inside a larger 3D volume (conceptually akin to tomograms).
	"""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)

	parser.add_argument("--apix",type=float,default=1.0,help="""Default=1.0. Sampling size of the output tomogram. Any models supplied via --input will be rescaled as needed.""")

	parser.add_argument("--box_size", type=int, default=64, help="""Default=64. Base longest-span for each object created (it can be modified from one object to another by other parameters).""")
	parser.add_argument("--background", type=float, default=None, help="""Default=None. Decimal between 0 and 1 for relative density of background to objects. Protein is ~40 percent denser than water; the default density for objects is 1, yielding ~0.71428571428 for water in comparison.""")

	parser.add_argument("--dilutionfactor", type=int, default=1, help="""Default=1. Alternative that supersedes --nptcls to determine the number of objects in the simulated final volume; 1=total crowdedness, the full volume is occupied with objects; 2=half of the volume is empty; 3=only one third if the volume has objects, etc.""") 
	
	parser.add_argument("--input", type=str, default=None, help="""Default=None. Explicitly list 3D image files to simulate particles inside the 'tomogram'; for example: groel.hdf,tric.hdf,mmcpn.hdf,hiv.hdf""")

	parser.add_argument("--half_shapes", type=str, default=None, help="""Default=None. Takes values "all" or "some". For example, --half_spheres=all. If "all", this will clip the generated shapes in half (for example, sphere's will be transformed into hemishperes, along x, y, or z, randomly assigned); if "half" is provided, only half of the shapes will be clipped in half.""")

	#parser.add_argument("--half_shapes_orthogonal", action="store_true", default=True, help="""Default=True. If on, this will clip the generated shapes in half from x, y, z axis only (one direction per shape, randomly chosen), as opposed to from random directions.""")

	parser.add_argument("--layers", type=int, default=None, help="""Default=None. Max value 4. If --layers=1, the objects will be hollow, if greater than 1, concentric layers with --layers number of "shells" to them (depending on the size of the objects, they may only accomodate 2-3 layers max).""")

	parser.add_argument("--ndiversity", type=int, default=1, help="""Default=1. Number of different shapes to include in simulated volume.""")
	parser.add_argument("--nptcls", type=int, default=None, help="""Default=None. Number of 64x64x64 shapes to include inside final simulated volume. The default is as many particles fit inside the volume v=tomonx*tomony*tomonz.""")

	parser.add_argument("--path", type=str, default='tomo_shapes',help="""Defautl=tomo_shapes. Directory to store results in. The default is a numbered series of directories containing the prefix 'tomo_shapes'; for example, tomo_shapes_02 will be the directory by default if 'tomo_shapes_01' already exists.""")

	parser.add_argument("--ppid", type=int, default=-1, help="""Default=-1. Set the PID of the parent process, used for cross platform PPID""")

	parser.add_argument("--savesteps", action="store_true", default=True, help="""Default=True. If on, intermediate files will be saved to --path.""")
	parser.add_argument("--shapes", type=str, default='all', help="""Default=all. Explicitly list desired shapes, separated by commas; for example: cube,prism,sphere. This overrides --ndiversity. Valid entries: all,cube,sphere,prism,pyramid,cylinder,disc,ellipsoid,cross2,cross3.""")
	parser.add_argument("--softedges", action="store_true", default=False, help="""Default=False. If on, the objects will have a soft instead of a sharp edge.""")

	parser.add_argument("--tomonx", type=int, default=512, help="""Default=512. Size in x for simiulated 'tomogarm'.""")
	parser.add_argument("--tomony", type=int, default=512, help="""Default=512. Size in y for simiulated 'tomogarm'.""")
	parser.add_argument("--tomonz", type=int, default=512, help="""Default=512. Size in z for simiulated 'tomogarm'.""")

	parser.add_argument("--varydensity", action="store_true", default=False, help="""Default=False. If on, different objects will have different density values.""")
	parser.add_argument("--varysize", action="store_true", default=False, help="""Default=False. If on, the objects will be simulated at different scales between 32*32*32 and 64*64*64 volumes.""")
	#parser.add_argument("--varyorientation", action="store_true", default=False, help="""Default=False. If on, the objects will be simulated in random orientations.""")

	parser.add_argument("--verbose", "-v", type=int, default=0, help="Default 0. Verbose level [0-9], higner number means higher level of verboseness.")
	
	parser.add_argument("--zc", type=int, default=None, help="""Default=None. This will fix all simulated objects to be centered in the z-plane zc.""")


	(options, args) = parser.parse_args()

	if options.half_shapes and options.half_shapes != 'all' and options.half_shapes != 'some':
		print("ERROR: half_shsapes takes values 'all' or 'some' not {}".format(options.half_shapes))
		sys.exit(1)

	#c:if the objects are layered, the 4th and smallest layer (inward) would be 9x9x9 boxels already for a 64x64x64 object. Can't go much smaller than that.
	if options.layers:
		if options.layers > 4:
			print("\nWARNING: maximum number of layers is 4; changing --layers={} to 4".format(options.layers)) 
			options.layers = 4

	#c:dictionary relating different shape types to integers; note that this is a dictionary of functions, to directly execute them when needed
	#c:there are 7 here, but it's easily expandable as new functions get created

	shapes_dict={'sheet':sheet, 'cube':cube, 'sphere':sphere, 'prism':prism, 'pyramid':pyramid, 'cylinder':cylinder, 'disc':disc, 'ellipsoid':ellipsoid, 'cross2':cross2, 'cross3':cross3, 'eman2_1':1, 'eman2_5':5}
	#shapes_dict={0:cube, 1:sphere, 2:prism, 3:pyramid, 4:cylinder, 5:disc, 6:ellipsoid, 7:cross2, 8:cross3}
	#shapes_dict_str={0:'cube', 1:'sphere', 2:'prism', 3:'pyramid', 4:'cylinder', 5:'disc', 6:'ellipsoid', 7:'cross2', 8:'cross3'}

	#c:make a directory where to store the output and temporary files
	makepath(options,stem='tomo_shapes')

	#c:log each execution of the script to an invisible log file at .eman2log.txt
	logger = E2init(sys.argv, options.ppid)

	#c:create a 'base' volume to be reshaped downstream; i.e., a box with value 1 for all voxels
	base_vol = EMData(options.box_size,options.box_size,options.box_size)
	base_vol.to_one()

	#c:randomly select --ndiversity different shapes from the shapes_dict above
	if options.ndiversity>12:
		options.ndiversity=12

	shape_ids = random.sample(shapes_dict.keys(), options.ndiversity)

	if options.verbose>8:
		print("\n--shapes={}".format(options.shapes))
		#sys.exit(1)
	
	if options.shapes:
		shape_ids=options.shapes.split(',')
		print("\nshape_ids={}".format(shape_ids))
		if 'all' in shape_ids:
			print("\nshapes_dict.keys()={}".format(shapes_dict.keys()))
			shape_ids = shapes_dict.keys()
			print("\nincluding ALL shape_ids={}".format(shape_ids))
		else:
			print("\n 'all' is NOT in shape_ids={}".format(shape_ids))

			

	if options.verbose>8:
		print("\nshape_ids={}".format(shape_ids))

	tomovol = options.tomonx*options.tomony*options.tomonz
	#c:ptclvol is not simply = options.box_size**3
	#c:since the objects can be rotated freely, we imagine them in a box circumscribing the sphere formed by the gyration or rotational average of the object.
	#c:the largest span would be the the hypothenus or diagonal of a options.boxsize^3 volume 
	side=options.box_size
	hyp=round(math.sqrt( 3 * (side**2) ))
	ptclvol=hyp**3
	
	nptclsmax = int( math.ceil(tomovol/ptclvol ) )
	if options.verbose>8:
		print("\nnptclsmax={}".format(nptclsmax))
	
	if options.dilutionfactor:
		print('\n--nptcls was reset from --nptlcs={}'.format(options.nptcls))
		options.nptcls = int(nptclsmax/options.dilutionfactor)
		print('\nto {}'.format(options.nptcls))

	print('\n--nptcls={}, nptclsmax={}'.format(options.nptcls,nptclsmax))
	if options.nptcls > nptclsmax:
		options.nptcls = nptclsmax
		print("\nWARNING: --nptcls is too high. The maximum number of box_size * box_size * box_size objects that fit in volume v={}*{}*{} is nptclsmax={}".format(options.tomonx,options.tomony,options.tomonz,nptclsmax))
	
	#c:generate all the points in a grid where to place each object; i.e., giving each object a box_size x box_size x box_size space within the final volume so that objects don't overlap
	coords=get_coords(options)

	if options.nptcls > len(coords):
		options.nptcls = len(coords)

	#c:generate an equal integer number of instances of each selected shape
	n_per_shape = int( math.ceil( options.nptcls/float(len(shape_ids))))
	if options.verbose:
		print("\ngenerating a total of n={} shapes, and n={} of each of t={} types".format(options.nptcls,n_per_shape,len(shape_ids)))

	#c:randomize the [x,y,x] coordinate elements of 'coords' to avoid placing the same types of objects clustered near each other, in downstream processes
	random.shuffle(coords)
	if options.verbose:
		print("\ngenerated a total of n={} shuffled coordinates".format(len(coords)))
	
	#c:loop over the selected shapes
	ptcl_count = 0 #c:explicitly count the particles since n_per_shape may not exactly divide --nptlcs, and math.ceil most likely will cause for options.nptcls to go over the nptcl limit
	loop_count = 0
	sym = Symmetries.get( 'c1' )
	
	blank_tomo = EMData(options.tomonx,options.tomony,options.tomonz)
	blank_tomo.to_zero()

	output_tomo = blank_tomo.copy()
	if options.background:
		output_tomo.to_one()
		output_tomo *= options.background

	box_size = options.box_size
	
	lines=[]
	print("\nthere are these many shape ids n={}, and they are={}".format(len(shape_ids),shape_ids))
	for j in shape_ids:
		print("\nexamining shape={}".format(j))

		output_label = EMData(options.tomonx,options.tomony,options.tomonz) #c: this will be a "label" in the segmentation sense, meaning a file with objects of the same class
		output_label.to_zero()

		shape = None
		base_vol_c = base_vol.copy() #c:make a fresh copy of the base volume each time we make a new shape type, since many operations downstream are applied 'in place'
		if 'eman2' not in j:
			if options.verbose:
				print("\ngenerating shape of type {}".format(j))
			shape = shapes_dict[j](options,base_vol_c)
			print("\nRETURNING *from* (sheet) fuction, type(shape)={}, shape[minimum]={}, shape[maximum]={}".format( type(shape), shape['minimum'], shape['maximum']) )

			if options.savesteps:
				shape.write_image(options.path+'/shapes_models.hdf',loop_count)
		elif 'eman2' in j:	#c:the 'processors' below binarize the pre-made images to a reasonable threshold, empirically
			if options.verbose:
				print("\nfetching pre-made shape of type {}".format(j))

			shape = test_image_3d(shapes_dict[j]).process("math.fft.resample",{"n":2}).process("threshold.binary",{"value":3.0}).process("mask.auto3d",{"nmaxseed":5,"radius":4})
			if options.savesteps:
				shape.write_image(options.path+'/shapes_models.hdf',loop_count)

		if options.layers:
			shape = gen_layers(options,shape)
			if options.verbose:
				print("\nadded layers to shape {}".format(j))


		if shape != None: #and ptcl_count < options.nptcls and ptcl_count < len(coords):
			print("\nworking with shape={}".format(j))
			print("\nshape[minimum]={}, shape[maximum]={}".format( type(shape), shape['minimum'], shape['maximum']) )
			print("\n\n\n\n\n******* n_per_shape={}".format(n_per_shape))
			
			for k in range(0,n_per_shape):
				print("\n\n\n\n\nFILTEEEEERRRRRRR")
				print("ptcl_count < options.nptcls = {}".format(ptcl_count < options.nptcls ))
				print("ptcl_count < len(coords) = {} ".format(ptcl_count < len(coords) ))

				if ptcl_count < options.nptcls and ptcl_count < len(coords):
					shape_c = shape.copy() #c:for orientations to mean anything (if needed for analyses later), they have to be defined with respect to the same "unrotated" frame of reference, or the unrotated shape

					if options.varydensity:
						intensity=random.uniform(0.9,1.1)
						shape_c*=intensity
						if options.verbose>8:
							print("\nvaried intensity by factor={} for ptcl={} of shape={}, global ptcl #={}".format(intensity,k,j,ptcl_count))
							print("\nVARYDENSITY shape_c[minimum]={}, shape_c[maximum]={}".format( type(shape_c), shape_c['minimum'], shape_c['maximum']) )

					if options.half_shapes:
						decision_factor = 0

						if options.half_shapes == 'all':
							decision_factor = 1
						elif options.half_shapes == 'some':
							decision_factor = k%2

						if decision_factor == 1:
							axis=random.choice(['x','y','z'])
							direction=random.choice([-1,1])
							if axis == 'x':
								if direction == -1:
									shape_c.process_inplace("mask.zeroedge3d",{"x0":box_size/2,"x1":0})
								elif direction == 1:
									shape_c.process_inplace("mask.zeroedge3d",{"x0":0,"x1":box_size/2})
							elif axis == 'y':
								if direction == -1:
									shape_c.process_inplace("mask.zeroedge3d",{"y0":box_size/2,"y1":0})
								elif direction == 1:
									shape_c.process_inplace("mask.zeroedge3d",{"y0":0,"y1":box_size/2})
							elif axis == 'z':
								if direction == -1:
									shape_c.process_inplace("mask.zeroedge3d",{"z0":box_size/2,"z1":0})
								elif direction == 1:
									shape_c.process_inplace("mask.zeroedge3d",{"z0":0,"z1":box_size/2})

					if options.varysize:
						shrink_factor = random.uniform(0.5,1.0)
						shape_c.process_inplace("xform.scale",{"scale":shrink_factor,"clip":box_size})

						if options.verbose>8:
							print("\nvaried size by factor={} for ptcl={} of shape={}, global ptcl #={}".format(shrink_factor,k,j,ptcl_count))
							print("\nVARYSIZE shape_c[minimum]={}, shape_c[maximum]={}".format( type(shape_c), shape_c['minimum'], shape_c['maximum']) )
					
					#c:apply a "transformation" (rotations) to the object only if requested and if the randomly-generated orientation is different from the identity (no rotation)
					orientation = Transform()
					#if options.varyorientation and j!='sphere': #shape type 1 is 'sphere', which looks the same in all orientations
					if j!='sphere': #shape type 1 is 'sphere', which looks the same in all orientations
						orientation = sym.gen_orientations("rand",{"n":1,"phitoo":1,"inc_mirror":1})[0]
					
					trans = box_size/4
					orientation.set_trans(random.randint(-1*trans, trans), random.randint(-1*trans,trans), random.randint(-1*trans,trans))

					if orientation != Transform():
						shape_c = clip3d(shape_c, int(math.ceil(box_size*math.sqrt(2))))
						shape_c.transform(orientation)
						
						if options.verbose>8:
							print("\napplied orientation t={} to ptcl={} of shape={}, global ptcl #={}".format(orientation,k,j,ptcl_count))
							print("\nTRANSFORMED shape_c[minimum]={}, shape_c[maximum]={}".format( type(shape_c), shape_c['minimum'], shape_c['maximum']) )

					
					#c:in the future, perhaps save orientations in case they're needed for subtomogram averaging tests later...???
					if options.verbose>8:
						print("\nptcl_count={}, len(coords)={}".format(ptcl_count,len(coords)))
					

					xc,yc,zc = int(coords[ptcl_count][0]),int(coords[ptcl_count][1]),int(coords[ptcl_count][2])

					if options.zc:
						zc = options.zc

					#c:will this cause overlap between objects?
					if options.dilutionfactor > 1:
						xc += random.randint(-options.box_size/2, options.box_size/2)
						yc += random.randint(-options.box_size/2, options.box_size/2)
						if not options.zc:
							zc += random.randint(-options.box_size/2, options.box_size/2)
					
					line = str(j)+'\t'+str(xc)+'\t'+str(yc)+'\t'+str(zc)
					if ptcl_count < options.nptcls - 1:
						line+='\n'
	
					lines.append(line)
					
					print("\n\n\nBEFORE INSERTION type(shape_c)={}, shape_c[minimum]={}, shape_c[maximum]={}".format( type(shape_c), shape_c['minimum'], shape_c['maximum']) )
					
					if options.savesteps:
						shape.write_image(options.path+'/'+j+'_models.hdf',k)
					
					output_label.insert_scaled_sum(shape_c,[xc,yc,zc])

					sys.stdout.flush()

					ptcl_count+=1
				
			
			#c:in case there are different densities across the label, add it to the tomogram before binarization
			output_tomo+=output_label

			#c:for the label to actually be a label it needs to be binarized into a map with 0s (voxels not belonging to the feature) and 1s (voxels belonging to the feature)
			
			#output_label.process_inplace("threshold.binary",{"value":0.01})
			output_label.process_inplace("threshold.binary",{"value":0.0})
			output_label *= -1
			output_label += 1
			output_label_file = options.path+'/label_'+str(j)+'.hdf'
			#output_label.write_image(output_label_file,0,EMUtil.get_image_ext_type("unknown"), False, None, 'int8', not(False))
			output_label.write_image(output_label_file,0)
			print("\nWrote output_label, type(output_label)={}, output_label[minimum]={}, output_label[maximum]={}".format( type(output_label), output_label['minimum'], output_label['maximum']) )


			#if options.verbose:
			print("\nbinarized label for shape={} and saved it to file={}".format(j,output_label_file))

			loop_count+=1

	output_tomo_file=options.path+'/simulated_tomogram.hdf'
	output_tomo_th = output_tomo.process("threshold.belowtominval",{"minval":options.background,"newval":options.background})

	maxval = 1.0
	if options.varydensity:
		maxval = 1.1
	output_tomo_th.process_inplace("threshold.clampminmax",{"maxval":maxval})
	
	output_tomo_th.write_image(output_tomo_file,0)	
	print("\nfinished simulating volume (the 'tomogram') and saved it to file={}".format(output_tomo_file))
	print("\ntype(output_tomo_th)={}, output_tomo_th[minimum]={}, output_tomo_th[maximum]={}".format( type(output_tomo_th), output_tomo_th['minimum'], output_tomo_th['maximum']) )

	
	with open(options.path+'/class_and_coords_file.txt','w') as f:
		f.writelines(lines)

	E2end(logger)

	elapsed = time.perf_counter() - start	
	print(str(timedelta(seconds=elapsed)))
	
	return


def gen_layers(options,shape):
	
	if options.verbose>8:
		print("\n(gen_layers) start")

	gr=(1+5**0.5)/2 #c:"golden" ratio to be used in calculating the shrinking factor that defines the realtive size between layers

	shape_layered = shape.copy()
	shape_full_size = shape['nx']
	for i in range(options.layers):
		shape_to_shrink=shape.copy()
		shape_shrunk=shape_to_shrink.process('math.fft.resample',{'n':gr**(i+1)})
		#shape_shrunk.process_inplace("threshold.binary",{'value':0.0})
		
		#c:invert the contrast of every other layer
		if i%2 == 0:
	
			shape_shrunk*=-1 

		shape_layered+= clip3d(shape_shrunk,shape_full_size)

	if options.background:
		background_inverse = 1.0/options.background
		shape_layered *= background_inverse

	return shape_layered


def get_coords(options):
	if options.verbose>8: print("\n(get_coords) start")

	#c:since the objects can be rotated freely, the coordinates need to be separated by the hypothenus or diagonal of a cube of side length = options.box_size to prevent overlaps between objects
	side=options.box_size
	
	hyp=round(math.sqrt( 3 * (side**2) )) #This is in 3D: sqrt(nx^2+ny^+nz^2), since nx=ny=nz=side


	xs = mid_points(options.tomonx,hyp,hyp)
	ys = mid_points(options.tomony,hyp,hyp)
	zs = mid_points(options.tomonz,hyp,hyp)

	if not xs or not ys or not zs:
		print("\nERROR: xs and/or ys and/or zs is empty")
		print("\n(get_coords) len(xs)={}\nlen(ys)={}\nlen(zs)={}".format(len(xs),len(ys),len(zs)))
		print("\n(get_coords) xs={}\nys={}\nzs={}".format(xs,ys,zs))
		sys.exit(1)


	return [ [xs[i],ys[j],zs[k]] for i in range(0,len(xs)) for j in range(0,len(ys)) for k in range(0,len(zs)) ]


#Returns the mid points of consecutive sections of size "segment" along the "length" of a line; "step" allows for overlaps
def mid_points(length,segment,step):
	#if options.verbose>8: print("\n(mid_points) start")
	return [int(round(p+segment/2.0)) for p in range(0,length,step) if (p+segment/2.0)<(length-(segment/2.0))]


def cube(options,vol_in):
	if options.verbose>8: print("\n(cube) start; type(vol_in)={}".format(type(vol_in)))
	
	#c:since the cubes might be rotated freely, it needs to be shrunk or masked to a size that will impede any density from going outside a 64^3 box when rotated
	side=vol_in['nx']

	#c:the radius of a circle circumscribed in a box is side/2
	radius=side/2.0

	#the side length of a cube within that circle is:
	new_side=math.sqrt(2*radius**2)
	
	diff=side-new_side

	return vol_in.process("mask.zeroedge3d",{"x0":diff/2.0,"x1":diff/2.0,"y0":diff/2.0,"y1":diff/2.0,"z0":diff/2.0,"z1":diff/2.0})


def sphere(options,vol_in):
	if options.verbose>8: print("\n(sphere) start; type(vol_in)={}".format(type(vol_in)))

	radius=vol_in['nx']/2.0
	return vol_in.process("mask.sharp",{"outer_radius":radius})


def sheet(options,vol_in):


	x0 = int(1)
	x1 = int(1)

	y0 = int(1)
	y1 = int(1)

	z0 = int(options.box_size/2 -1)
	z1 = int(options.box_size/2 -1)
	
	if options.verbose>8: 
		print("\n(sheet), type(vol_in)={}, vol_in[minimum]={}, vol_in[maximum]={}".format( type(vol_in), vol_in['minimum'], vol_in['maximum']) )
		print("\nmasking zeroedge3d with values x0={}, x1={}, y0={}, y1={}, z0={}, z1={}".format(x0,x1,y0,y1,z0,z1) )

	return vol_in.process("mask.zeroedge3d",{"x0":x0,"x1":x1,"y0":y0,"y1":y1,"z0":z0,"z1":z1})

	
def prism(options,vol_in,thickness=None):
	if options.verbose>8: print("\n(prism) start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	thick_to_use=vol_in['nx']/gr
	if thickness!=None:
		thick_to_use=thickness

	to_erase=(vol_in['nx']-thick_to_use)/2.0
	return vol_in.process("mask.zeroedge3d",{"x0":to_erase,"x1":to_erase,"y0":to_erase,"y1":to_erase})


def pyramid(options,vol_in):
	if options.verbose>8: print("\n(pyramid) start; type(vol_in)={}".format(type(vol_in)))

	pyramid_angle = 63.4

	#c:as for the cube, the side-length of the pyramid base sides and its height needs to be scaled so that upon rotation, no densities go outside the box
	old_box=vol_in['nx']
	radius=old_box/2.0
	box=math.sqrt(2*radius**2)

	#c:this is just a trick to use a bigger "negative" box rotated and translated by precise amounts, 4 times, to "carve" each side of a pyramid out of vol_in 
	box_expanded=3*box

	subtractor=vol_in.copy()
	subtractor_expanded = clip3d(subtractor,box_expanded)
	proto_pyramid = subtractor_expanded.copy()

	subtractor_expanded_scaled=subtractor_expanded.process("xform.scale",{"scale":3,"clip":box_expanded})
	subtractor_expanded_scaled_neg=-1*subtractor_expanded_scaled

	tz=2.15*box*math.cos(math.radians(pyramid_angle-90)) #c:This should put the edge face of a slanted negative cube thrice in side length as vol_in to intersetc with the bottom left corner of vol_in 
	print("\ntz={}".format(tz))

	tslant = Transform({"type":"eman","az":0,"alt":pyramid_angle,"phi":0,"tz":tz})
	subtractor_expanded_slanted=subtractor_expanded_scaled_neg.copy()
	subtractor_expanded_slanted.transform(tslant)

	proto_pyramid_steps=[]
	subtractors=[]
	
	for i in range(0,4):
		az=i*90
		t1 = Transform({"type":"eman","az":0,"alt":90})
		t2 = Transform({"type":"eman","az":az,"alt":-90,"phi":0})
		ttot = t2*t1
		subtractor_i = subtractor_expanded_slanted.copy()
		subtractor_i.transform(ttot)
		subtractors.append(subtractor_i)
		proto_pyramid+=subtractor_i
		proto_pyramid_steps.append(proto_pyramid)
		proto_pyramid.process_inplace("threshold.belowtozero",{"minval":0.0})

		sys.stdout.flush()

	pyramid=clip3d(proto_pyramid,box)

	return pyramid


def cylinder(options,vol_in):
	if options.verbose>8: print("\n(cylinder)start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	height=vol_in['nx']
	radius=round(height/gr**2)
	return vol_in.process("testimage.cylinder",{"height":height,"radius":radius})


def disc(options,vol_in):
	if options.verbose>8: print("\n(disc) start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	major=vol_in['nx']/2.0
	minor=int(round(major/gr))
	height=int(round(minor/gr))
	return vol_in.process("testimage.disc",{"major":major,"minor":minor,"height":height})


def ellipsoid(options,vol_in):
	if options.verbose>8: print("\n(ellipsoid) start; type(vol_in)={}".format(type(vol_in)))
	vol_in.to_zero()
	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	a=vol_in['nx']/2.0
	b=round(a/gr)
	c=round(b/gr)
	return vol_in.process("testimage.ellipsoid",{"a":a,"b":b,"c":c,"fill":1})


def cross2(options,vol_in,thickness=None):
	if options.verbose>8: print("\n(cross2) start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	thick_to_use=vol_in['nx']/gr**3
	if thickness!=None:
		thick_to_use=thickness
	
	element1 = prism(options,vol_in,thick_to_use)
	
	element2 = element1.copy()
	t=Transform({"type":"eman","az":0,"alt":90,"phi":0})
	element2.transform(t)

	return element1+element2


def cross3(options,vol_in,thickness=None):
	if options.verbose>8: print("\n(cross3) start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	thick_to_use=vol_in['nx']/gr**3
	if thickness!=None:
		thick_to_use=thickness
	
	element1 = prism(options,vol_in,thick_to_use)
	
	element2 = element1.copy()
	t=Transform({"type":"eman","az":0,"alt":90,"phi":0})
	element2.transform(t)

	element3 = element1.copy()
	t=Transform({"type":"eman","az":0,"alt":90,"phi":90})
	element3.transform(t)

	return element1+element2+element3


if __name__ == "__main__":
    main()
    sys.stdout.flush()'
	'''