#!/usr/bin/env python3
"""
tomo_intensity_scaler.py

Author: Jesús G. Galaz-Montoya
Created: 2021
Last modification: 2025-04-21

Make all voxel values positive using one of three methods:
  1) --threshold: zero out negative values
  2) --linear (default): shift and scale to [0,1]
  3) --abs: take absolute value of voxels

Tracks progress with tqdm (if verbose>=1), and logs messages based on --verbose level.
"""

import os
import sys
import time
from EMAN2 import EMArgumentParser, EMUtil, EMData, E2init, E2end, EMANVERSION
import concurrent.futures
from multiprocessing import cpu_count
from tqdm import tqdm


def parse_args():
    parser = EMArgumentParser(usage=__doc__, version=EMANVERSION)

    parser.add_argument(
        "--abs",
        action="store_true",
        default=False,
        help="Default=False. Take the absolute value of all voxels."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Default=None. Path to an image file or directory of images to scale."
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        default=False,
        help="Default=False. Linearly scale data to [0,1] range."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Default=False. Overwrite original files with scaled output."
    )
    parser.add_argument(
        "--ppid",
        type=int,
        default=-1,
        help="Default=-1. Set parent PID for logging."
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        default=False,
        help="Default=False. Zero out any negative voxel values."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=cpu_count(),
        help=f"Default={cpu_count()}. Number of parallel worker processes."
    )
    parser.add_argument(
        "--verbose", "-v",
        type=int,
        default=0,
        help="Default=0. Verbosity level; higher values print more feedback."
    )

    options, _ = parser.parse_args()
    if not options.input:
        parser.error("An --input file or directory must be specified.")
    return options


def main():
    start_time = time.perf_counter()
    options = parse_args()
    logger = E2init(sys.argv, options.ppid)

    if os.path.isdir(options.input):
        files = [
            os.path.join(options.input, f)
            for f in sorted(os.listdir(options.input))
            if f.lower().endswith((".hdf", ".mrc"))
        ]
        if options.verbose >= 1:
            print(f"Found {len(files)} files to process in '{options.input}'.")
        process_directory(options, files)
    else:
        proj = 'proj' in os.path.basename(options.input).lower()
        process_file(options, options.input, 0, proj)

    E2end(logger)
    elapsed = time.perf_counter() - start_time
    print(f"\nFinished in {elapsed:.2f} seconds.")


def process_directory(options, files):
    """
    Process multiple files in parallel using ProcessPoolExecutor and tqdm.
    """
    workers = options.threads or cpu_count()
    if options.verbose >= 1:
        print(f"Using {workers} parallel workers")

    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_file, options, f, idx, 'proj' in f.lower()): f
            for idx, f in enumerate(files)
        }
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            disable=(options.verbose == 0),
            desc="Processing files"
        ):
            pass
    elapsed = time.perf_counter() - start
    if options.verbose >= 1:
        print(f"Parallel processing finished in {elapsed:.2f} seconds.")


def process_file(options, filepath, count, proj):
    """
    Read one image file (or HDF stack), apply scaling, and write output.
    """
    if options.verbose >= 2:
        print(f"[{count}] Processing '{filepath}' (proj={proj})")

    base, ext = os.path.splitext(filepath)
    n_images = EMUtil.get_image_count(filepath)
    output = base + '_scaled' + ext
    if options.overwrite:
        output = filepath

    if n_images == 1 and not proj:
        img = EMData(filepath, 0)
        img_out = apply_scaling(img, options)
        img_out.write_image(output)
    else:
        for i in range(n_images):
            img = EMData(filepath, i)
            img_out = apply_scaling(img, options)
            img_out.write_image(output, i)

    if options.verbose >= 2:
        print(f"[{count}] Wrote output: '{output}'")
    return count


def apply_scaling(img, options):
    """
    Apply the selected scaling method to an EMData object.
    """
    # 1) threshold: zero out negatives
    if options.threshold:
        img_out = img.copy()
        img_out.process_inplace('threshold.clampmin', {'minval': 0})
        return img_out
    # 2) absolute: square then sqrt
    if options.abs:
        img_sq = img * img
        img_abs = img_sq.process('math.sqrt', {})
        return img_abs
    # 3) linear (default)
    return scale_zero_to_one(img)


def scale_zero_to_one(img):
    """
    Linearly scale voxel values so that minimum→0 and maximum→1.
    """
    minv = img['minimum']
    maxv = img['maximum']
    if maxv == minv:
        return img - minv  # empties to zeros
    return (img - minv) / (maxv - minv)


if __name__ == '__main__':
    main()
