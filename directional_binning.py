
#!/usr/bin/env python
#====================
#Author: Jesus Galaz-Montoya - 014/Jan/2025, Last update: 01/2025
#====================

import argparse
import numpy as np
from EMAN2 import *
import os
from scipy.ndimage import zoom

def bin_data_fractional(data, binx, biny, binz):
    zoom_factors = (1 / binz, 1 / biny, 1 / binx)  # Correct shrinkage direction
    return zoom(data, zoom_factors, order=1)  # Linear interpolation
    
def mrc_to_eman(input_file, output_file, binx, biny, binz, make_isotropic):
    img = EMData(input_file)
    voxel_size_x, voxel_size_y, voxel_size_z = img['apix_x'], img['apix_y'], img['apix_z']
    
    if make_isotropic:
        binx, biny, binz = compute_isotropic_bins(voxel_size_x, voxel_size_y, voxel_size_z)
    
    data = img.numpy().astype(np.float32)
    binned_data = bin_data_fractional(data, binx, biny, binz)
    
    # Convert binned data back to EMData
    binned_img = EMNumPy.numpy2em(binned_data)
    binned_img.set_attr_dict(img.get_attr_dict())  # Copy all header attributes

    # Correct voxel sizes (apix) by multiplying with the binning factors
    binned_img['apix_x'] = voxel_size_x * binx
    binned_img['apix_y'] = voxel_size_y * biny
    binned_img['apix_z'] = voxel_size_z * binz

    # **CRITICAL FIX**: Explicitly update MRC-specific header fields
    binned_img.set_attr("MRC.xlen", binned_img['apix_x'] * binned_data.shape[2])  # X-axis
    binned_img.set_attr("MRC.ylen", binned_img['apix_y'] * binned_data.shape[1])  # Y-axis
    binned_img.set_attr("MRC.zlen", binned_img['apix_z'] * binned_data.shape[0])  # Z-axis

    binned_img.write_image(output_file)

def hdf_to_eman(input_file, output_file, binx, biny, binz, make_isotropic):
    img = EMData(input_file)
    voxel_size_x, voxel_size_y, voxel_size_z = img['apix_x'], img['apix_y'], img['apix_z']
    
    if make_isotropic:
        binx, biny, binz = compute_isotropic_bins(voxel_size_x, voxel_size_y, voxel_size_z)
    
    data = img.numpy().astype(np.float32)
    binned_data = bin_data_fractional(data, binx, biny, binz)
    
    binned_img = EMNumPy.numpy2em(binned_data)
    binned_img.set_attr_dict(img.get_attr_dict())  # Copy all header attributes

    # **FIXED**: Multiply voxel size to reflect binning increase
    binned_img['apix_x'] = voxel_size_x * binx
    binned_img['apix_y'] = voxel_size_y * biny
    binned_img['apix_z'] = voxel_size_z * binz

    # **NEW**: Update the physical dimensions
    binned_img['xlen'] = binned_img['apix_x'] * binned_data.shape[2]
    binned_img['ylen'] = binned_img['apix_y'] * binned_data.shape[1]
    binned_img['zlen'] = binned_img['apix_z'] * binned_data.shape[0]

    binned_img.write_image(output_file)


def compute_isotropic_bins(apix_x, apix_y, apix_z):
    max_apix = max(apix_x, apix_y, apix_z)
    binx = max_apix / apix_x
    biny = max_apix / apix_y
    binz = max_apix / apix_z
    return binx, biny, binz

def main():
    parser = argparse.ArgumentParser(description="Bin and convert MRC or HDF images independently along each axis.")
    parser.add_argument('input_file', type=str, help='Path to the input MRC or HDF file')
    parser.add_argument('output_file', type=str, help='Path to the output MRC or HDF file')
    parser.add_argument('--binx', type=float, default=1, help='Binning factor along X-axis')
    parser.add_argument('--biny', type=float, default=1, help='Binning factor along Y-axis')
    parser.add_argument('--binz', type=float, default=1, help='Binning factor along Z-axis')
    parser.add_argument('--make_isotropic', action='store_true', help='Automatically bin to make voxel sizes isotropic')
    args = parser.parse_args()

    input_ext = os.path.splitext(args.input_file)[1].lower()
    output_ext = os.path.splitext(args.output_file)[1].lower()

    if input_ext in ['.mrc', '.hdf'] and output_ext in ['.mrc', '.hdf']:
        mrc_to_eman(args.input_file, args.output_file, args.binx, args.biny, args.binz, args.make_isotropic)
    else:
        print("Unsupported file format conversion. Supported conversions: MRC<->HDF.")

if __name__ == '__main__':
    main()

