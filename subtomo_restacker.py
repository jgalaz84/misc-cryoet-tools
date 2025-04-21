#!/usr/bin/env python
"""
subtomo_restacker.py
Created on April 21, 2025 by Jesús G. Galaz-Montoya, Ph.D.
Last modified on April 21, 2025 by Jesús G. Galaz-Montoya, Ph.D.

A script to restack an EMAN2-compatible HDF stack by including/excluding
specified indices, extracting ranges and subsets, optionally randomizing order,
and writing to a new HDF file. Includes a progress bar, timing, and command logging.
"""

import argparse
import logging
import os
import random
import sys
import time
from EMAN2 import EMData, EMUtil
from tqdm import tqdm


def parse_indices_list(arg):
    """Parse comma-separated string of integers into a sorted list of unique ints."""
    if not arg:
        return []
    try:
        return sorted(set(int(x) for x in arg.split(',') if x.strip()))
    except ValueError:
        raise argparse.ArgumentTypeError("Index lists must be comma-separated integers")


def main():
    # Record start time
    start_time = time.time()

    # Set up argument parser with parameters in alphabetical order
    parser = argparse.ArgumentParser(
        description="Restack an EMAN2-compatible HDF stack with include/exclude, range, subset, and randomize options."
    )
    parser.add_argument('--exclude', default='',
                        help='default: ""; comma-separated list of indices to exclude from the input HDF stack.')
    parser.add_argument('--first', type=int, default=None,
                        help='default: None; first index (inclusive) to include in the stack range.')
    parser.add_argument('--include', default='',
                        help='default: ""; comma-separated list of indices to include from the input HDF stack.')
    parser.add_argument('--input', required=True,
                        help='default: None; path to the input EMAN2-compatible HDF stack.')
    parser.add_argument('--last', type=int, default=None,
                        help='default: None; last index (inclusive) to include in the stack range.')
    parser.add_argument('--output', default=None,
                        help='default: None; path to output HDF stack, if not set appends "_restacked" to input filename.')
    parser.add_argument('--randomize_order', action='store_true', default=False,
                        help='default: False; randomize the order of the output stack after selection.')
    parser.add_argument('--subset', type=int, default=None,
                        help='default: None; size of a random subset to extract after selection.')
    parser.add_argument('--verbose', type=int, choices=[0,1,2,3], default=0,
                        help='default: 0; verbosity level: 0=no output, 1=errors, 2=info, 3=debug.')

    args = parser.parse_args()

    # Configure logging
    log_levels = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.INFO, 3: logging.DEBUG}
    logging.basicConfig(level=log_levels.get(args.verbose, logging.CRITICAL),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        if '.' in args.input:
            base, ext = args.input.rsplit('.', 1)
            output_file = f"{base}_restacked.{ext}"
        else:
            output_file = f"{args.input}_restacked"

    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file: {output_file}")

    # Log the command for reproducibility
    cmd_log = ' '.join(sys.argv)
    log_filename = "subtomo_restacker.log"
    try:
        with open(log_filename, 'a') as log_f:
            log_f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {cmd_log}\n")
    except Exception as e:
        logging.error(f"Failed to write command log: {e}")

    # Remove existing output to start fresh
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            logging.debug(f"Removed existing file {output_file}")
        except Exception as e:
            logging.error(f"Could not remove existing output file: {e}")

    # Get total number of images
    total_images = EMUtil.get_image_count(args.input)
    logging.debug(f"Total images in stack: {total_images}")

    # Build initial list of indices (0-based)
    all_indices = list(range(total_images))

    # Apply include/exclude filters
    include_idxs = parse_indices_list(args.include)
    exclude_idxs = parse_indices_list(args.exclude)
    if include_idxs:
        selected = [i for i in include_idxs if 0 <= i < total_images]
    else:
        selected = all_indices.copy()
    if exclude_idxs:
        selected = [i for i in selected if i not in exclude_idxs]
    logging.debug(f"Selected indices after include/exclude: {selected}")

    # Apply first/last range filters
    if args.first is not None:
        selected = [i for i in selected if i >= args.first]
    if args.last is not None:
        selected = [i for i in selected if i <= args.last]
    logging.debug(f"Selected indices after first/last: {selected}")

    # Apply subset selection
    if args.subset is not None and args.subset < len(selected):
        selected = random.sample(selected, args.subset)
        logging.debug(f"Selected indices after subset of size {args.subset}: {selected}")

    # Apply randomize_order if requested
    if args.randomize_order:
        random.shuffle(selected)
        logging.debug("Randomized order of selected indices")

    if not selected:
        logging.error("No indices selected for restacking. Exiting.")
        sys.exit(1)

    # Loop through selected indices, read and write each image
    for out_idx, in_idx in enumerate(tqdm(selected, desc="Restacking", unit="image")):
        try:
            img = EMData(args.input, in_idx)
            img.write_image(output_file, out_idx)
            logging.debug(f"Wrote image {out_idx} (source index {in_idx})")
        except Exception as e:
            logging.error(f"Failed writing index {in_idx}: {e}")
            sys.exit(1)

    logging.info(f"Written {len(selected)} images to {output_file}")

    # Print timing information
    elapsed = time.time() - start_time
    print(f"Completed restacking in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()
