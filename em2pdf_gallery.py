#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya, 03/2023; last update 07/2024
#
# Modifications:
#  - 03/2025:
#    (1) Added margins so text is not cut off.
#    (2) Default --input_dir = '.'.
#    (3) Default --output_dir = 'em2pdf_gallery' with numbered suffixes (_00, _01, etc.).
#    (4) Added logging of runs in a log file.
#    (5) Restored --equalize functionality (CLAHE), storing data as 16-bit,
#        with 8-bit normalization only for PDF display.
#    (6) Introduced iterative equalization for projections that automatically
#        adjusts CLAHE parameters with pre-clamping and optional log transform.
#    (7) Added --prj_method to choose the projection method (sum, mean, median),
#        --log_transform to apply a logarithmic transformation,
#        and --clamp_sigma to control the clamping threshold.
#    (8) Saves histogram plots for each projection as PNG files.
#    (9) Added --regularize to remove spike pixels in the final 8-bit projection.
#   (10) Updated remove_spike_pixels to handle >= 250, improved histogram title, etc.

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from EMAN2 import *
from EMAN2_utils import *
import datetime
import sys
import math
from skimage import exposure  # for CLAHE-based histogram equalization
import matplotlib.pyplot as plt

###############################################################################
# Post-processing: Remove Spike Pixels
###############################################################################
def remove_spike_pixels(final_8bit, intensity_cutoff=250, spike_threshold=0.05):
    """
    If more than spike_threshold fraction of pixels >= intensity_cutoff,
    reassign those pixels with random values from the rest of the distribution.
    
    Parameters:
      final_8bit : np.ndarray (uint8)
          The 8-bit image to be post-processed.
      intensity_cutoff : int
          Pixel values >= this cutoff are considered "spike" (default: 250).
      spike_threshold : float
          Fraction threshold above which we consider the spike problematic (default: 0.05).
    
    Returns:
      final_8bit : np.ndarray (uint8)
          The image with spike pixels replaced.
    """
    fraction_spike = np.mean(final_8bit >= intensity_cutoff)
    print(f"DEBUG: fraction_spike >= {intensity_cutoff} = {fraction_spike:.4f}")
    if fraction_spike > spike_threshold:
        mask = (final_8bit >= intensity_cutoff)
        valid_pixels = final_8bit[~mask]
        if len(valid_pixels) < 1:
            final_8bit[mask] = 128
            print("DEBUG: All pixels were spikes; replaced with mid-gray (128).")
        else:
            random_values = np.random.choice(valid_pixels, size=np.count_nonzero(mask))
            final_8bit[mask] = random_values
            print(f"DEBUG: Replaced {np.count_nonzero(mask)} spike pixels.")
    return final_8bit

###############################################################################
# Iterative Equalization for Projections with Pre-Clamping and Optional Log Transform
###############################################################################
def clamp_extremes(image_16, sigma=3):
    """
    Clamp pixel values that exceed mean ± sigma*std.
    Assumes image_16 is a 16-bit image scaled to [0, 65535].
    Returns the clamped and re-normalized 16-bit image.
    """
    img_float = image_16.astype(np.float64) / 65535.0
    mean_val = np.mean(img_float)
    std_val = np.std(img_float)
    lower_bound = mean_val - sigma * std_val
    upper_bound = mean_val + sigma * std_val
    img_clamped = np.clip(img_float, lower_bound, upper_bound)
    img_rescaled = (img_clamped - lower_bound) / (upper_bound - lower_bound)
    return (img_rescaled * 65535.0).astype(np.uint16)

def iterative_equalize_prj(
    image_data,
    clip_limit_start=0.01,
    clip_limit_min=0.0005,
    kernel_size_start=64,
    saturation_threshold=0.01,
    max_iterations=5,
    clamp_sigma=3,
    log_transform=False
):
    """
    Iteratively apply CLAHE to a 2D projection image, with pre-clamping to remove extreme values,
    and optionally a logarithmic transformation. Adjusts clip_limit (and kernel_size) until the fraction
    of saturated pixels (i.e. pixels at 65535) is below saturation_threshold.

    Parameters:
      image_data : np.ndarray
         The 2D projection image.
      clip_limit_start : float
         Initial clip limit for CLAHE.
      clip_limit_min : float
         Minimum clip limit allowed.
      kernel_size_start : int
         Initial kernel (tile) size.
      saturation_threshold : float
         Maximum acceptable fraction of saturated pixels.
      max_iterations : int
         Maximum iterations to adjust parameters.
      clamp_sigma : float
         Sigma threshold for clamping extremes.
      log_transform : bool
         If True, apply a logarithmic transformation before equalization.

    Returns:
      eq_16 : np.ndarray (uint16)
         The final equalized image (16-bit).
      final_clip : float
         The final clip limit used.
      final_kernel_size : int
         The final kernel size used.
    """
    # Convert to 16-bit if necessary.
    if image_data.dtype != np.uint16:
        if np.max(image_data) == 0:
            image_16 = image_data.astype(np.uint16)
        else:
            image_16 = (image_data.astype(np.float64) / np.max(image_data)) * 65535.0
            image_16 = image_16.astype(np.uint16)
    else:
        image_16 = image_data

    # Optionally apply logarithmic transformation
    if log_transform:
        img_log = np.log1p(image_16.astype(np.float64))
        img_log = (img_log - img_log.min()) / (img_log.max() - img_log.min())
        image_16 = (img_log * 65535.0).astype(np.uint16)

    # Pre-clamp extremes
    image_16 = clamp_extremes(image_16, sigma=clamp_sigma)

    def fraction_saturated(img):
        return np.count_nonzero(img == 65535) / img.size

    clip_limit = clip_limit_start
    kernel_size = kernel_size_start
    eq_16 = image_16.copy()

    for iteration in range(max_iterations):
        eq_float = exposure.equalize_adapthist(
            image_16, clip_limit=clip_limit, kernel_size=kernel_size
        )
        eq_16 = (eq_float * 65535.0).astype(np.uint16)
        if fraction_saturated(eq_16) <= saturation_threshold:
            break
        else:
            clip_limit /= 2.0
            if clip_limit < clip_limit_min:
                clip_limit = clip_limit_min
            kernel_size = int(kernel_size * 1.25)
            if kernel_size < 3:
                kernel_size = 3
            if kernel_size % 2 == 0:
                kernel_size += 1

    return eq_16, clip_limit, kernel_size

###############################################################################
# Utility Functions
###############################################################################
def create_labeled_image(image_data, label_filename, index_str, font_small):
    """
    Convert a single-channel (L) NumPy array to RGB, then draw the filename on one line,
    and the index on a second line, using a smaller font.
    """
    image = Image.fromarray(image_data, mode='L').convert('RGB')
    draw = ImageDraw.Draw(image)
    text_x = 10
    text_y = 10
    draw.text((text_x, text_y), label_filename, fill=(255, 255, 255), font=font_small)
    text_y += 20
    draw.text((text_x, text_y), f"Index: {index_str}", fill=(255, 255, 255), font=font_small)
    return image

def normalize_image(image_data):
    """
    Normalize image data to 8-bit range [0..255].
    """
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    if max_val > min_val:
        image_data = (image_data - min_val) / (max_val - min_val) * 255
    else:
        image_data = np.zeros_like(image_data)
    return image_data.astype('uint8')

def numpy_to_emdata(numpy_array):
    em_data = from_numpy(numpy_array.astype(np.float32))
    return em_data

def save_eman2_stack(images, output_path):
    for i, img in enumerate(images):
        em_data = numpy_to_emdata(img)
        em_data.write_image(output_path, i)
    print(f"Saved stack to {output_path}")

def ensure_unique_directory(base_dir):
    counter = 0
    new_dir = f"{base_dir}_{counter:02d}"
    while os.path.exists(new_dir):
        counter += 1
        new_dir = f"{base_dir}_{counter:02d}"
    os.makedirs(new_dir)
    return new_dir

###############################################################################
# Main Program
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="""Create a gallery of 2D cryoEM images from an input directory or stack
                       and output a PDF gallery. Optionally perform histogram equalization on projections,
                       with options for projection method, log transformation, clamping threshold, and spike regularization."""
    )
    parser.add_argument("--clamp_sigma", type=float, default=3.0, help="Clamping threshold in sigma (default: 3)")
    parser.add_argument("--equalize", action='store_true', help="Perform 16-bit CLAHE-based histogram equalization on projections (skip slices)")
    parser.add_argument("--input_dir", type=str, default='.', help="Directory containing input files (default: current directory)")
    parser.add_argument("--input_string", type=str, default='', help="String that should be contained in files to process")
    parser.add_argument("--log_transform", action='store_true', help="Apply logarithmic transformation before equalization")
    parser.add_argument("--output_dir", type=str, default='em2pdf_gallery', help="Base directory to save outputs (default: em2pdf_gallery). Numbered suffixes (_00, _01, etc.) are appended.")
    parser.add_argument("--output_file", type=str, required=True, help="Name of PDF file to save image gallery")
    parser.add_argument("--prj_method", type=str, choices=["sum", "mean", "median"], default="sum", help="Projection method to use: sum, mean, or median")
    parser.add_argument("--regularize", action='store_true', help="Apply post-processing to remove spike pixels from projections")
    parser.add_argument("--save_histogram", action='store_true', help="Save histogram plots for each projection")
    parser.add_argument("--save_prjs", action='store_true', help="Save projections to HDF stack")
    parser.add_argument("--save_slices", action='store_true', help="Save slices to HDF stack")
    parser.add_argument("--spike_threshold", type=float, default=0.05, help="Fraction threshold for spike removal")

    args = parser.parse_args()

    output_dir = ensure_unique_directory(args.output_dir)
    print(f"Output directory: {output_dir}")

    log_file = os.path.join(output_dir, "em2pdf_gallery.log")
    with open(log_file, 'a') as lf:
        lf.write(f"\n---\nTime: {datetime.datetime.now()}\n")
        lf.write(f"Command: {' '.join(sys.argv)}\n")

    files = [f for f in os.listdir(args.input_dir) if args.input_string in f and os.path.splitext(f)[-1] in extensions()]
    files.sort()
    print(f"Files to process: {files}")

    images = []
    projections = []
    slices_ = []

    try:
        try:
            font_small = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font_small = ImageFont.load_default()

        for f in files:
            file_path = os.path.join(args.input_dir, f)
            n = EMUtil.get_image_count(file_path)
            # Assign fname here so it's available regardless of saving histograms.
            fname = os.path.splitext(f)[0]
            print(f"\nProcessing file {f} which has {n} images in it")
            for i in range(n):
                em_image = EMData(file_path, i)
                image_data = em_image.numpy()

                if image_data.ndim == 3:
                    mid_index = image_data.shape[0] // 2
                    raw_middle_slice = image_data[mid_index].copy()

                    if args.prj_method == "sum":
                        raw_projection = image_data.sum(axis=0).copy()
                    elif args.prj_method == "mean":
                        raw_projection = image_data.mean(axis=0).copy()
                    elif args.prj_method == "median":
                        raw_projection = np.median(image_data, axis=0).copy()

                    middle_slice_8 = normalize_image(raw_middle_slice)
                    slices_.append(middle_slice_8)

                    if args.equalize:
                        eq_16, final_clip, final_ks = iterative_equalize_prj(
                            raw_projection,
                            clip_limit_start=0.01,
                            clip_limit_min=0.0005,
                            kernel_size_start=64,
                            saturation_threshold=0.01,
                            max_iterations=5,
                            clamp_sigma=args.clamp_sigma,
                            log_transform=args.log_transform
                        )
                        display_projection = normalize_image(eq_16)
                        projections.append(eq_16)
                    else:
                        display_projection = normalize_image(raw_projection)
                        projections.append(display_projection)

                    if args.regularize:
                        display_projection = remove_spike_pixels(
                            display_projection,
                            intensity_cutoff=250,
                            spike_threshold=args.spike_threshold
                        )

                    if args.save_histogram:
                        hist_path = os.path.join(output_dir, f"{fname}_prj_{i}_hist.png")
                        plt.figure(figsize=(8,6))
                        plt.hist(display_projection.ravel(), bins=256, range=(0,255), color='gray')
                        plt.title(f"{fname}\nProjection Index {i}")
                        plt.xlabel("Intensity")
                        plt.ylabel("Frequency")
                        plt.savefig(hist_path)
                        plt.close()
                        print(f"Saved histogram to {hist_path}")

                    middle_slice_image = create_labeled_image(middle_slice_8, fname, str(i), font_small)
                    projection_image = create_labeled_image(display_projection, fname, str(i), font_small)

                    margin_x = 20
                    margin_y = 40
                    combined_width = middle_slice_image.width + projection_image.width + (margin_x * 3)
                    combined_height = max(middle_slice_image.height, projection_image.height) + margin_y
                    combined_image = Image.new('RGB', (combined_width, combined_height))
                    combined_image.paste(middle_slice_image, (margin_x, 0))
                    combined_image.paste(projection_image, (margin_x + middle_slice_image.width, 0))

                    draw = ImageDraw.Draw(combined_image)
                    title = f"{fname}\nIndex: {i}"
                    text_position = (margin_x, combined_height - 30)
                    draw.text(text_position, title, (255, 255, 255), font=font_small)

                    images.append(combined_image)

                else:
                    # Process 2D images similarly...
                    if args.equalize:
                        eq_16, final_clip, final_ks = iterative_equalize_prj(
                            image_data,
                            clip_limit_start=0.01,
                            clip_limit_min=0.0005,
                            kernel_size_start=64,
                            saturation_threshold=0.01,
                            max_iterations=5,
                            clamp_sigma=args.clamp_sigma,
                            log_transform=args.log_transform
                        )
                        slices_.append(eq_16)
                        display_8 = normalize_image(eq_16)
                    else:
                        norm_img = normalize_image(image_data)
                        slices_.append(norm_img)
                        display_8 = norm_img

                    if args.regularize:
                        display_8 = remove_spike_pixels(display_8, 250, args.spike_threshold)

                        single_image = create_labeled_image(display_8, fname, str(i), font_small)

                        margin_x = 20
                        margin_y = 60
                        text_height = 30
                        new_width = single_image.width + (margin_x * 2)
                        new_height = single_image.height + margin_y

                        combined_image = Image.new('RGB', (new_width, new_height))
                        combined_image.paste(single_image, (margin_x, 0))

                        draw = ImageDraw.Draw(combined_image)
                        title = f"{fname}\nIndex: {i}"
                        text_position = (margin_x, new_height - text_height)
                        draw.text(text_position, title, (255, 255, 255), font=font_small)

                        images.append(combined_image)
                    else:
                        # 2D image
                        if args.equalize:
                            eq_16, final_clip, final_ks = iterative_equalize_prj(
                                image_data,
                                clip_limit_start=0.01,
                                clip_limit_min=0.0005,
                                kernel_size_start=64,
                                saturation_threshold=0.01,
                                max_iterations=5,
                                clamp_sigma=args.clamp_sigma,
                                log_transform=args.log_transform
                            )
                            slices_.append(eq_16)
                            display_8 = normalize_image(eq_16)
                        else:
                            norm_img = normalize_image(image_data)
                            slices_.append(norm_img)
                            display_8 = norm_img

                        # Optionally regularize 2D images as well (if desired).
                        if args.regularize:
                            display_8 = remove_spike_pixels(display_projection, 250, args.spike_threshold)

                        fname = os.path.splitext(f)[0]
                        single_image = create_labeled_image(display_8, fname, str(i), font_small)

                        margin_x = 20
                        margin_y = 60
                        text_height = 30
                        new_width = single_image.width + (margin_x * 2)
                        new_height = single_image.height + margin_y

                        combined_image = Image.new('RGB', (new_width, new_height))
                        combined_image.paste(single_image, (margin_x, 0))

                        draw = ImageDraw.Draw(combined_image)
                        title = f"{fname}\nIndex: {i}"
                        text_position = (margin_x, new_height - text_height)
                        draw.text(text_position, title, (255, 255, 255), font=font_small)

                        images.append(combined_image)

        # Save combined images to a PDF.
        if images:
            pdf_output_path = os.path.join(output_dir, args.output_file)
            images[0].save(pdf_output_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
            print(f"Saved PDF to {pdf_output_path}")
        else:
            print("No images found to save in the PDF.")

        # Optionally save stacks.
        if args.save_prjs and projections:
            save_eman2_stack(projections, os.path.join(output_dir, 'projections.hdf'))
        if args.save_slices and slices_:
            save_eman2_stack(slices_, os.path.join(output_dir, 'slices.hdf'))

        with open(log_file, 'a') as lf:
            lf.write("Status: success\n")

    except Exception as e:
        with open(log_file, 'a') as lf:
            lf.write(f"Status: fail\nReason: {str(e)}\n")
        raise e

if __name__ == '__main__':
    main()