#!/usr/bin/env python3
# Author: [Your Name]
# Date: 2024-09-03
# Description: This script uses the EMAN2 library to read an HDF or MRC file containing multiple images or slices,
#              counts the total number of images or slices, and then separates them into two new files:
#              one containing images or slices at even indices and the other containing those at odd indices.

from EMAN2 import *
from EMAN2_utils import *

def split_even_odd_stacks(input_filename):
    # Determine the file type and set appropriate output filenames
    if input_filename.endswith('.hdf'):
        file_type = 'hdf'
        even_filename = input_filename.replace('.hdf', '_even.hdf')
        odd_filename = input_filename.replace('.hdf', '_odd.hdf')
        
        # Get the number of images in the input HDF file
        num_images = EMUtil.get_image_count(input_filename)
    
    elif input_filename.endswith('.mrc'):
        file_type = 'mrc'
        even_filename = input_filename.replace('.mrc', '_even.mrc')
        odd_filename = input_filename.replace('.mrc', '_odd.mrc')
        
        # Get the number of slices in the MRC file using the nz parameter
        temp_image = EMData(input_filename, 0, True)  # Read header only
        num_images = temp_image.get_attr('nz')
    
    else:
        print("Unsupported file type. Only HDF and MRC formats are supported.")
        return
    
    # Initialize counters for even and odd images/slices
    even_count = 0
    odd_count = 0

    if file_type == 'hdf':
        # Loop through all images in the input HDF file
        for i in range(num_images):
            # Read the i-th image from the input file
            image = EMData(input_filename, i)

            # Check if the index is even or odd and write to the respective file
            if i % 2 == 0:
                image.write_image(even_filename, even_count)
                even_count += 1
            else:
                image.write_image(odd_filename, odd_count)
                odd_count += 1

    elif file_type == 'mrc':
        # For MRC, extract slices using Region and save them as even and odd stacks
        for i in range(num_images):
            # Define the region to extract one slice along the Z-axis
            region = Region(0, 0, i, temp_image['nx'], temp_image['ny'], 1)
            slice_image = EMData(input_filename, 0, False, region)
            
            # Check if the slice index is even or odd and write to the respective file
            if i % 2 == 0:
                slice_image.write_image(even_filename, even_count)
                even_count += 1
            else:
                slice_image.write_image(odd_filename, odd_count)
                odd_count += 1

    print(f"Split completed: {even_count} even images/slices written to {even_filename}, {odd_count} odd images/slices written to {odd_filename}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python split_hdf_even_odd.py <input_filename>")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    split_even_odd_stacks(input_filename)
