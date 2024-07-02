#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya, 03/2023; last update 07/2024

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from EMAN2 import *
from EMAN2_utils import *

def create_labeled_image(image_data, label, font):
    image = Image.fromarray(image_data, mode='L').convert('RGB')
    draw = ImageDraw.Draw(image)
    text_position = (10, 10)
    draw.text(text_position, label, (255, 255, 255), font=font)
    return image

def normalize_image(image_data):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    if max_val > min_val:  # Avoid division by zero
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
    counter = 1
    new_dir = base_dir
    while os.path.exists(new_dir):
        new_dir = f"{base_dir}_{counter}"
        counter += 1
    os.makedirs(new_dir)
    return new_dir

def main():
    parser = argparse.ArgumentParser(description="""Create a gallery of 2D cryoEM images in the current directory or from an input stack with 
        multiple 2D images and prints them to a PDF file.""")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files")
    parser.add_argument("--input_string", type=str, required=False, default='', help="String that should be contained in files to process")
    parser.add_argument("--output_file", type=str, required=True, help="Name of PDF file to save image gallery")
    parser.add_argument("--output_dir", type=str, default='.', help="Directory to save outputs")
    parser.add_argument("--save_prjs", action='store_true', help="Save projections to HDF stack")
    parser.add_argument("--save_slices", action='store_true', help="Save slices to HDF stack")

    args = parser.parse_args()

    output_dir = ensure_unique_directory(args.output_dir)
    print(f"Output directory: {output_dir}")
    files = [f for f in os.listdir(args.input_dir) if args.input_string in f and os.path.splitext(f)[-1] in extensions()]
    files.sort()
    print(f"Files to process: {files}")
    
    images = []
    projections = []
    slices = []

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for f in files:
        file_path = os.path.join(args.input_dir, f)
        n = EMUtil.get_image_count(file_path)
        print(f"\nProcessing file {f} which has {n} images in it")

        for i in range(n):
            image = EMData(file_path, i)
            image_data = image.numpy()
            
            if image_data.ndim == 3:
                middle_index = image_data.shape[0] // 2
                middle_slice = image_data[middle_index]
                projection = image_data.sum(axis=0)

                middle_slice = normalize_image(middle_slice)
                projection = normalize_image(projection)

                slices.append(middle_slice)
                projections.append(projection)

                middle_slice_image = create_labeled_image(middle_slice, "Middle Z-Slice", font)
                projection_image = create_labeled_image(projection, "Z Reprojection", font)

                combined_width = middle_slice_image.width + projection_image.width
                combined_height = max(middle_slice_image.height, projection_image.height) + 40
                combined_image = Image.new('RGB', (combined_width, combined_height))
                combined_image.paste(middle_slice_image, (0, 0))
                combined_image.paste(projection_image, (middle_slice_image.width, 0))

                draw = ImageDraw.Draw(combined_image)
                text_position = (10, combined_height - 30)
                title = f"File: {f} - Index: {i}"
                draw.text(text_position, title, (255, 255, 255), font=font)

                images.append(combined_image)

    if images:
        pdf_output_path = os.path.join(output_dir, args.output_file)
        images[0].save(pdf_output_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
        print(f"Saved PDF to {pdf_output_path}")
    else:
        print("No images found to save in the PDF.")

    if args.save_prjs and projections:
        save_eman2_stack(projections, os.path.join(output_dir, 'projections.hdf'))
    
    if args.save_slices and slices:
        save_eman2_stack(slices, os.path.join(output_dir, 'slices.hdf'))

if __name__ == '__main__':
    main()