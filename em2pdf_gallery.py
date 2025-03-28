#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya, 03/2023; last update 03/28/2025
#
# Modifications:
#  - Ensures index numbers are truly in the bottom-right corner of each image/subimage.
#  - If --subtomograms is provided, prj-only and slices-only PDFs are laid out in a grid.
#  - The filename only appears on the first page for each file in prj-only and slices-only PDFs.
#  - The index is drawn with a dark background and white text for readability.

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from EMAN2 import *
from EMAN2_utils import *
import datetime
import sys
import math
import time
from skimage import exposure  # for CLAHE-based histogram equalization
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################################################
# Helper: Draw text with a background rectangle
###############################################################################
def draw_text_with_bg(draw, position, text, font, text_color=(255,255,255), bg_color=(50,50,50), padding=2):
    """
    Draw text with a small background rectangle behind it for readability.
    """
    bbox = draw.textbbox((0,0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x, y = position
    draw.rectangle([x, y, x+w+padding*2, y+h+padding*2], fill=bg_color)
    draw.text((x+padding, y+padding), text, font=font, fill=text_color)

###############################################################################
# Remove Spike Pixels
###############################################################################
def remove_spike_pixels(final_8bit, intensity_cutoff=250, spike_threshold=0.05):
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
# Iterative Equalization for Projections
###############################################################################
def clamp_extremes(image_16, sigma=3):
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
    if image_data.dtype != np.uint16:
        if np.max(image_data) == 0:
            image_16 = image_data.astype(np.uint16)
        else:
            image_16 = (image_data.astype(np.float64) / np.max(image_data)) * 65535.0
            image_16 = image_16.astype(np.uint16)
    else:
        image_16 = image_data

    if log_transform:
        img_log = np.log1p(image_16.astype(np.float64))
        img_log = (img_log - img_log.min()) / (img_log.max() - img_log.min())
        image_16 = (img_log * 65535.0).astype(np.uint16)

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
def normalize_image(image_data):
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

def center_position(page_width, page_height, img_width, img_height):
    x = (page_width - img_width) // 2
    y = (page_height - img_height) // 2
    return x, y

def add_page_numbers(pages):
    n_pages = len(pages)
    if n_pages < 1:
        return pages
    
    try:
        font_page = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font_page = ImageFont.load_default()
    
    for i, page_img in enumerate(pages, start=1):
        draw_pg = ImageDraw.Draw(page_img)
        page_text = f"{i}/{n_pages}"
        bbox_pg = draw_pg.textbbox((0,0), page_text, font=font_page)
        pw = bbox_pg[2] - bbox_pg[0]
        ph = bbox_pg[3] - bbox_pg[1]
        draw_pg.text((page_img.width - pw - 10, page_img.height - ph - 10),
                     page_text, fill=(0, 0, 0), font=font_page)
    return pages

###############################################################################
# Draw index in the bottom-right corner of a 2D array
###############################################################################
def label_index_in_bottom_right(image_array, index_val):
    """
    Convert array to RGB, then place the index in the *true* bottom-right corner,
    with a small margin (5px).
    """
    img = Image.fromarray(image_array)
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw_img = ImageDraw.Draw(img)
    try:
        font_index = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font_index = ImageFont.load_default()

    index_text = str(index_val)
    bbox = draw_img.textbbox((0,0), index_text, font=font_index)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    # Place so the bounding box is 5px from the right and bottom edges
    x_pos = img.width - text_w - 5
    y_pos = img.height - text_h - 5
    draw_text_with_bg(draw_img, (x_pos, y_pos), index_text, font_index)

    return img

###############################################################################
# Grid layout with a two-phase header: first page vs subsequent pages
###############################################################################
def create_grid_pages_multiheader(
    image_list,
    page_size,
    margin,
    first_page_header,
    subsequent_page_header
):
    """
    Creates a grid layout. The first page uses first_page_header, subsequent pages
    use subsequent_page_header. Each subimage is already labeled with an index if needed.
    """
    pages = []
    if not image_list:
        return pages

    page_width, page_height = page_size
    orig_w, orig_h = image_list[0].size
    scale = 1.0
    if orig_w + 2*margin > page_width or orig_h + 2*margin > page_height:
        scale = min((page_width - 2*margin) / orig_w, (page_height - 2*margin) / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    n_cols = max(1, (page_width - 2*margin) // (new_w + margin))
    n_rows = max(1, (page_height - 100) // (new_h + margin))
    images_per_page = n_cols * n_rows

    try:
        font_header = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font_header = ImageFont.load_default()

    for i in range(0, len(image_list), images_per_page):
        page = Image.new('RGB', (page_width, page_height), color=(255,255,255))
        draw = ImageDraw.Draw(page)
        # Decide header
        if i == 0:
            header_text = first_page_header
        else:
            header_text = subsequent_page_header
        bbox = draw.textbbox((0,0), header_text, font=font_header)
        header_w = bbox[2] - bbox[0]
        header_h = bbox[3] - bbox[1]
        header_x = (page_width - header_w) // 2
        draw.text((header_x, 20), header_text, fill=(0,0,0), font=font_header)
        
        total_w = n_cols * new_w + (n_cols - 1) * margin
        total_h = n_rows * new_h + (n_rows - 1) * margin
        offset_x = (page_width - total_w) // 2
        offset_y = 20 + header_h + 20

        page_imgs = image_list[i:i+images_per_page]
        for j, img in enumerate(page_imgs):
            if scale != 1.0:
                img_resized = img.resize((new_w, new_h), Image.ANTIALIAS)
            else:
                img_resized = img
            col = j % n_cols
            row = j // n_cols
            x = offset_x + col * (new_w + margin)
            y = offset_y + row * (new_h + margin)
            page.paste(img_resized, (x, y))
        pages.append(page)
    return pages

###############################################################################
# Single-image page
###############################################################################
def create_single_image_page(page_size, header_text, img, index):
    """
    One image per page. We rely on the function that labels the index in the bottom-right.
    """
    page = Image.new('RGB', page_size, color=(255,255,255))
    draw = ImageDraw.Draw(page)
    try:
        font_header = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font_header = ImageFont.load_default()
    
    header_margin = 20
    bbox = draw.textbbox((0,0), header_text, font=font_header)
    header_w = bbox[2] - bbox[0]
    header_h = bbox[3] - bbox[1]
    header_x = (page_size[0] - header_w) // 2
    draw.text((header_x, header_margin), header_text, fill=(0,0,0), font=font_header)
    
    available_top = header_margin + header_h + 10
    available_height = page_size[1] - available_top - 30
    
    # The 'img' is presumably already labeled if desired, but we can label again if needed.
    # We'll trust the label_index_in_bottom_right() function for that.
    # Or we can do it here. We'll just place it as-is, assuming 'img' is already labeled.
    img_x, img_y = center_position(page_size[0], available_height, img.width, img.height)
    page.paste(img, (img_x, available_top + img_y))
    return page

###############################################################################
# Gallery page: slice–prj pairs
###############################################################################
def create_gallery_page(page_size, header_text, slice_img, prj_img, index):
    """
    One pair (slice, prj) per page, each labeled with index in bottom-right corner
    of each subimage.
    """
    page = Image.new('RGB', page_size, color=(255,255,255))
    draw = ImageDraw.Draw(page)
    try:
        font_header = ImageFont.truetype("arial.ttf", 24)
        font_index = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font_header = ImageFont.load_default()
        font_index = ImageFont.load_default()
    
    header_margin = 20
    bbox = draw.textbbox((0,0), header_text, font=font_header)
    header_w = bbox[2] - bbox[0]
    header_h = bbox[3] - bbox[1]
    header_x = (page_size[0] - header_w) // 2
    draw.text((header_x, header_margin), header_text, fill=(0,0,0), font=font_header)
    
    available_top = header_margin + header_h + 10
    available_height = page_size[1] - available_top - 30
    
    # Convert slices to labeled images
    slice_img_labeled = label_index_in_bottom_right(np.array(slice_img), index)
    prj_img_labeled   = label_index_in_bottom_right(np.array(prj_img), index)

    margin_between = 20
    combined_w = slice_img_labeled.width + prj_img_labeled.width + margin_between
    combined_h = max(slice_img_labeled.height, prj_img_labeled.height)
    
    combined = Image.new('RGB', (combined_w, combined_h), color=(255,255,255))
    slice_x = 0
    prj_x = slice_img_labeled.width + margin_between
    combined.paste(slice_img_labeled, (slice_x, 0))
    combined.paste(prj_img_labeled, (prj_x, 0))
    
    comb_x, comb_y = center_position(page_size[0], available_height, combined_w, combined_h)
    page.paste(combined, (comb_x, available_top + comb_y))
    return page

###############################################################################
# Main Program
###############################################################################
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="""Create a gallery of 2D or 3D cryoEM images from an input directory or stack
                       and output a PDF gallery. Optionally perform histogram equalization on projections,
                       with options for projection method, log transform, clamping threshold, spike regularization,
                       and layout optimizations for subtomograms.
                    """
    )
    parser.add_argument("--clamp_sigma", type=float, default=3.0, help="Clamping threshold in sigma (default: 3)")
    parser.add_argument("--equalize", action='store_true', help="Perform 16-bit CLAHE-based histogram equalization on projections (skip slices)")
    parser.add_argument("--input_dir", type=str, default='.', help="Directory containing input files (default: current directory)")
    parser.add_argument("--input_string", type=str, default='', help="String that should be contained in files to process")
    parser.add_argument("--log_transform", action='store_true', help="Apply logarithmic transformation before equalization")
    parser.add_argument("--output_dir", type=str, default='em2pdf_gallery', help="Base directory to save outputs (default: em2pdf_gallery). Numbered suffixes (_00, _01, etc.) are appended.")
    parser.add_argument("--output_file", type=str, default="gallery_slice_prj_pairs.pdf", help="Name of PDF file to save image gallery (default: gallery_slice_prj_pairs.pdf)")
    parser.add_argument("--prj_method", type=str, choices=["sum", "mean", "median"], default="sum", help="Projection method to use: sum, mean, or median")
    parser.add_argument("--regularize", action='store_true', help="Apply post-processing to remove spike pixels from projections")
    parser.add_argument("--save_histogram", action='store_true', help="Save histogram plots for each projection as PNG files")
    parser.add_argument("--save_prjs", action='store_true', help="Save projections to HDF stack")
    parser.add_argument("--save_slices", action='store_true', help="Save slices to HDF stack")
    parser.add_argument("--spike_threshold", type=float, default=0.05, help="Fraction threshold for spike removal (only used with --regularize)")
    parser.add_argument("--subtomograms", action='store_true', default=False, help="If set, optimize layout for subtomograms (multiple images per PDF page)")
    parser.add_argument("--directions", type=str, default="z", help="Comma-separated directions (e.g., 'x,y,z') to process (default: z)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode enabled.")

    directions = [d.strip().lower() for d in args.directions.split(",") if d.strip()]
    if len(directions) == 0:
        directions = ['z']

    output_dir = ensure_unique_directory(args.output_dir)
    if args.verbose:
        print(f"Output directory: {output_dir}")

    log_file = os.path.join(output_dir, "em2pdf_gallery.log")
    with open(log_file, 'a') as lf:
        lf.write(f"\n---\nTime: {datetime.datetime.now()}\n")
        lf.write(f"Command: {' '.join(sys.argv)}\n")

    files = [f for f in os.listdir(args.input_dir)
             if args.input_string in f and os.path.splitext(f)[-1] in extensions()]
    files.sort()
    if args.verbose:
        print(f"Files to process: {files}")

    # Data structures
    galleries = {d: {} for d in directions}
    prj_images = {d: {} for d in directions}
    slice_images = {d: {} for d in directions}

    try:
        # 1) Read images from each file
        for f in tqdm(files, desc="Processing files"):
            file_path = os.path.join(args.input_dir, f)
            n = EMUtil.get_image_count(file_path)
            fname = os.path.splitext(f)[0]
            if args.verbose:
                print(f"\nProcessing file {f} with {n} images.")
            for i in range(n):
                em_image = EMData(file_path, i)
                image_data = em_image.numpy()

                if image_data.ndim == 3:
                    for d in directions:
                        if d == 'z':
                            mid_index = image_data.shape[0] // 2
                            slice_raw = image_data[mid_index].copy()
                            if args.prj_method == "sum":
                                prj_raw = image_data.sum(axis=0).copy()
                            elif args.prj_method == "mean":
                                prj_raw = image_data.mean(axis=0).copy()
                            elif args.prj_method == "median":
                                prj_raw = np.median(image_data, axis=0).copy()
                        elif d == 'y':
                            mid_index = image_data.shape[1] // 2
                            slice_raw = image_data[:, mid_index, :].copy()
                            if args.prj_method == "sum":
                                prj_raw = image_data.sum(axis=1).copy()
                            elif args.prj_method == "mean":
                                prj_raw = image_data.mean(axis=1).copy()
                            elif args.prj_method == "median":
                                prj_raw = np.median(image_data, axis=1).copy()
                        elif d == 'x':
                            mid_index = image_data.shape[2] // 2
                            slice_raw = image_data[:, :, mid_index].copy()
                            if args.prj_method == "sum":
                                prj_raw = image_data.sum(axis=2).copy()
                            elif args.prj_method == "mean":
                                prj_raw = image_data.mean(axis=2).copy()
                            elif args.prj_method == "median":
                                prj_raw = np.median(image_data, axis=2).copy()
                        else:
                            continue

                        slice_norm = normalize_image(slice_raw)
                        if args.equalize:
                            eq_16, _, _ = iterative_equalize_prj(
                                prj_raw,
                                clip_limit_start=0.01,
                                clip_limit_min=0.0005,
                                kernel_size_start=64,
                                saturation_threshold=0.01,
                                max_iterations=5,
                                clamp_sigma=args.clamp_sigma,
                                log_transform=args.log_transform
                            )
                            prj_disp = normalize_image(eq_16)
                        else:
                            prj_disp = normalize_image(prj_raw)

                        if args.regularize:
                            prj_disp = remove_spike_pixels(prj_disp, intensity_cutoff=250,
                                                           spike_threshold=args.spike_threshold)

                        if args.save_histogram:
                            hist_path = os.path.join(output_dir, f"{fname}_{d}_prj_{i}_hist.png")
                            plt.figure(figsize=(8,6))
                            plt.hist(prj_disp.ravel(), bins=256, range=(0,255), color='gray')
                            plt.title(f"{fname}\n{d}-projection Index {i}")
                            plt.xlabel("Intensity")
                            plt.ylabel("Frequency")
                            plt.savefig(hist_path)
                            plt.close()
                            if args.verbose:
                                print(f"Saved histogram to {hist_path}")

                        galleries[d].setdefault(fname, []).append((i, slice_norm, prj_disp))
                        prj_images[d].setdefault(fname, []).append((i, prj_disp))
                        slice_images[d].setdefault(fname, []).append((i, slice_norm))

                else:
                    # 2D
                    norm_img = normalize_image(image_data)
                    if args.equalize:
                        eq_16, _, _ = iterative_equalize_prj(
                            image_data,
                            clip_limit_start=0.01,
                            clip_limit_min=0.0005,
                            kernel_size_start=64,
                            saturation_threshold=0.01,
                            max_iterations=5,
                            clamp_sigma=args.clamp_sigma,
                            log_transform=args.log_transform
                        )
                        norm_img = normalize_image(eq_16)
                    if args.regularize:
                        norm_img = remove_spike_pixels(norm_img, 250, args.spike_threshold)
                    # For 2D, treat as 'z'
                    galleries['z'].setdefault(fname, []).append((i, norm_img, norm_img))
                    prj_images['z'].setdefault(fname, []).append((i, norm_img))
                    slice_images['z'].setdefault(fname, []).append((i, norm_img))

        page_size = (850, 1100)

        #######################################################################
        # 2) Gallery PDFs (slice–prj pairs)
        #######################################################################
        for d in directions:
            pages = []
            for fname in sorted(galleries[d].keys()):
                all_pairs = galleries[d][fname]
                for i, (idx, slice_img, prj_img) in enumerate(all_pairs):
                    if i == 0:
                        header_text = f"{fname}  {d}-slice  {d}-prj"
                    else:
                        header_text = f"{d}-slice  {d}-prj"
                    page = create_gallery_page(page_size,
                                               header_text,
                                               Image.fromarray(slice_img),
                                               Image.fromarray(prj_img),
                                               idx)
                    pages.append(page)
            pages = add_page_numbers(pages)
            if pages:
                out_name = f"{d}_gallery_slice_prj_pairs.pdf"
                out_path = os.path.join(output_dir, out_name)
                pages[0].save(out_path, "PDF", resolution=100.0,
                              save_all=True, append_images=pages[1:])
                if args.verbose:
                    print(f"Saved gallery PDF for direction {d} to {out_path}")

        #######################################################################
        # 3) Projection-only PDFs
        #######################################################################
        for d in directions:
            final_pages = []
            for fname in sorted(prj_images[d].keys()):
                labeled_imgs = []
                for (idx, array_) in prj_images[d][fname]:
                    # Label each array with its index in the bottom-right
                    labeled_imgs.append(label_index_in_bottom_right(array_, idx))

                if len(labeled_imgs) < 1:
                    continue

                if args.subtomograms:
                    first_header = f"{d}-prj  {fname}"
                    subsequent_header = f"{d}-prj"
                    pages_for_file = create_grid_pages_multiheader(
                        labeled_imgs,
                        page_size,
                        margin=10,
                        first_page_header=first_header,
                        subsequent_page_header=subsequent_header
                    )
                    final_pages.extend(pages_for_file)
                else:
                    # Single-image approach
                    for (idx, array_) in prj_images[d][fname]:
                        # The array is re-labeled so the index is visible
                        # but we can also do it again in single_image_page
                        labeled_img = label_index_in_bottom_right(array_, idx)
                        header_text = f"{d}-prj  {fname}  (index {idx})"
                        single_page = create_single_image_page(
                            page_size,
                            header_text,
                            labeled_img,
                            idx
                        )
                        final_pages.append(single_page)

            final_pages = add_page_numbers(final_pages)
            if final_pages:
                out_name = f"{d}_prj.pdf"
                out_path = os.path.join(output_dir, out_name)
                final_pages[0].save(out_path, "PDF", resolution=100.0,
                                    save_all=True, append_images=final_pages[1:])
                if args.verbose:
                    print(f"Saved prj PDF for direction {d} to {out_path}")

        #######################################################################
        # 4) Slice-only PDFs
        #######################################################################
        for d in directions:
            final_pages = []
            for fname in sorted(slice_images[d].keys()):
                labeled_imgs = []
                for (idx, array_) in slice_images[d][fname]:
                    labeled_imgs.append(label_index_in_bottom_right(array_, idx))

                if len(labeled_imgs) < 1:
                    continue

                if args.subtomograms:
                    first_header = f"{d}-slice  {fname}"
                    subsequent_header = f"{d}-slice"
                    pages_for_file = create_grid_pages_multiheader(
                        labeled_imgs,
                        page_size,
                        margin=10,
                        first_page_header=first_header,
                        subsequent_page_header=subsequent_header
                    )
                    final_pages.extend(pages_for_file)
                else:
                    for (idx, array_) in slice_images[d][fname]:
                        labeled_img = label_index_in_bottom_right(array_, idx)
                        header_text = f"{d}-slice  {fname}  (index {idx})"
                        single_page = create_single_image_page(
                            page_size,
                            header_text,
                            labeled_img,
                            idx
                        )
                        final_pages.append(single_page)

            final_pages = add_page_numbers(final_pages)
            if final_pages:
                out_name = f"{d}_slices.pdf"
                out_path = os.path.join(output_dir, out_name)
                final_pages[0].save(out_path, "PDF", resolution=100.0,
                                    save_all=True, append_images=final_pages[1:])
                if args.verbose:
                    print(f"Saved slices PDF for direction {d} to {out_path}")

        #######################################################################
        # 5) Orthogonal PDFs (if x,y,z are all provided)
        #######################################################################
        if set(directions) == set(['x','y','z']):
            # Orthogonal prj
            ortho_prj_pages = []
            num_volumes = 0
            if 'x' in prj_images:
                for fname in sorted(prj_images['x'].keys()):
                    num_volumes = max(num_volumes, len(prj_images['x'][fname]))
            for i in range(num_volumes):
                page = Image.new('RGB', page_size, color=(255,255,255))
                draw = ImageDraw.Draw(page)
                try:
                    font_header = ImageFont.truetype("arial.ttf", 24)
                    font_sub = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font_header = ImageFont.load_default()
                    font_sub = ImageFont.load_default()
                header_text = "Orthogonal prj"
                bbox = draw.textbbox((0,0), header_text, font=font_header)
                header_w = bbox[2] - bbox[0]
                header_h = bbox[3] - bbox[1]
                header_x = (page_size[0] - header_w) // 2
                draw.text((header_x, 20), header_text, fill=(0,0,0), font=font_header)

                directions_xyz = ['x','y','z']
                sub_imgs = []
                for dd in directions_xyz:
                    if dd not in prj_images or not prj_images[dd]:
                        continue
                    first_file = sorted(prj_images[dd].keys())[0]
                    if i >= len(prj_images[dd][first_file]):
                        continue
                    idx_val, array_ = prj_images[dd][first_file][i]
                    # Label direction letter top-left, index bottom-right
                    sub_img = label_index_in_bottom_right(array_, idx_val)
                    # Then add direction letter top-left
                    draw_sub = ImageDraw.Draw(sub_img)
                    draw_text_with_bg(draw_sub, (5,5), dd, font_sub)
                    sub_imgs.append(sub_img)

                margin_between = 10
                total_w = sum(im.width for im in sub_imgs) + margin_between*(len(sub_imgs)-1)
                max_h = max(im.height for im in sub_imgs) if sub_imgs else 0
                combined = Image.new('RGB', (total_w, max_h), color=(255,255,255))
                x_offset = 0
                for im in sub_imgs:
                    off_y = center_position(im.width, max_h, im.width, im.height)[1]
                    combined.paste(im, (x_offset, off_y))
                    x_offset += im.width + margin_between

                available_top = 20 + header_h + 10
                comb_x, comb_y = center_position(page_size[0], page_size[1]-available_top,
                                                 combined.width, combined.height)
                page.paste(combined, (comb_x, available_top+comb_y))
                ortho_prj_pages.append(page)
            ortho_prj_pages = add_page_numbers(ortho_prj_pages)
            if ortho_prj_pages:
                out_path = os.path.join(output_dir, "orthogonal_prj.pdf")
                ortho_prj_pages[0].save(out_path, "PDF", resolution=100.0,
                                        save_all=True, append_images=ortho_prj_pages[1:])
                if args.verbose:
                    print(f"Saved orthogonal prj PDF to {out_path}")

            # Orthogonal slices
            ortho_slice_pages = []
            num_volumes_slices = 0
            if 'x' in slice_images:
                for fname in sorted(slice_images['x'].keys()):
                    num_volumes_slices = max(num_volumes_slices, len(slice_images['x'][fname]))
            for i in range(num_volumes_slices):
                page = Image.new('RGB', page_size, color=(255,255,255))
                draw = ImageDraw.Draw(page)
                try:
                    font_header = ImageFont.truetype("arial.ttf", 24)
                    font_sub = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font_header = ImageFont.load_default()
                    font_sub = ImageFont.load_default()
                header_text = "Orthogonal slices"
                bbox = draw.textbbox((0,0), header_text, font=font_header)
                header_w = bbox[2] - bbox[0]
                header_h = bbox[3] - bbox[1]
                header_x = (page_size[0] - header_w) // 2
                draw.text((header_x, 20), header_text, fill=(0,0,0), font=font_header)

                directions_xyz = ['x','y','z']
                sub_imgs = []
                for dd in directions_xyz:
                    if dd not in slice_images or not slice_images[dd]:
                        continue
                    first_file = sorted(slice_images[dd].keys())[0]
                    if i >= len(slice_images[dd][first_file]):
                        continue
                    idx_val, array_ = slice_images[dd][first_file][i]
                    sub_img = label_index_in_bottom_right(array_, idx_val)
                    draw_sub = ImageDraw.Draw(sub_img)
                    draw_text_with_bg(draw_sub, (5,5), dd, font_sub)
                    sub_imgs.append(sub_img)

                margin_between = 10
                total_w = sum(im.width for im in sub_imgs) + margin_between*(len(sub_imgs)-1)
                max_h = max(im.height for im in sub_imgs) if sub_imgs else 0
                combined = Image.new('RGB', (total_w, max_h), color=(255,255,255))
                x_offset = 0
                for im in sub_imgs:
                    off_y = center_position(im.width, max_h, im.width, im.height)[1]
                    combined.paste(im, (x_offset, off_y))
                    x_offset += im.width + margin_between

                available_top = 20 + header_h + 10
                comb_x, comb_y = center_position(page_size[0], page_size[1]-available_top,
                                                 combined.width, combined.height)
                page.paste(combined, (comb_x, available_top+comb_y))
                ortho_slice_pages.append(page)
            ortho_slice_pages = add_page_numbers(ortho_slice_pages)
            if ortho_slice_pages:
                out_path = os.path.join(output_dir, "orthogonal_slices.pdf")
                ortho_slice_pages[0].save(out_path, "PDF", resolution=100.0,
                                          save_all=True, append_images=ortho_slice_pages[1:])
                if args.verbose:
                    print(f"Saved orthogonal slices PDF to {out_path}")

        #######################################################################
        # Optionally save HDF stacks
        #######################################################################
        if args.save_prjs:
            for d in prj_images:
                all_prjs = []
                for fname in prj_images[d]:
                    for (idx, data) in prj_images[d][fname]:
                        all_prjs.append(data)
                if all_prjs:
                    save_eman2_stack(all_prjs, os.path.join(output_dir, f"{d}_prj_stack.hdf"))
        if args.save_slices:
            for d in slice_images:
                all_slices = []
                for fname in slice_images[d]:
                    for (idx, data) in slice_images[d][fname]:
                        all_slices.append(data)
                if all_slices:
                    save_eman2_stack(all_slices, os.path.join(output_dir, f"{d}_slices_stack.hdf"))

        with open(log_file, 'a') as lf:
            lf.write("Status: success\n")

        elapsed = time.time() - start_time
        print(f"Total runtime: {elapsed:.2f} seconds.")

    except Exception as e:
        with open(log_file, 'a') as lf:
            lf.write(f"Status: fail\nReason: {str(e)}\n")
        raise e

if __name__ == '__main__':
    main()


'''
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
    parser.add_argument("--spike_threshold", type=float, default=0.05, help="Requires --regularize. Fraction threshold for spike removal.")

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
    '''