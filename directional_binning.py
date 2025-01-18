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

    # **FIXED**: Multiply voxel size to reflect binning increase
    binned_img['apix_x'] = voxel_size_x * binx
    binned_img['apix_y'] = voxel_size_y * biny
    binned_img['apix_z'] = voxel_size_z * binz

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


'''
#!/usr/bin/env python
from past.utils import old_div
from builtins import range
#====================
#Author: Jesus Galaz-Montoya - 14/Jan/2025, Last update: 17/Jan/2025
#====================

import argparse
import numpy as np
import mrcfile
import h5py
import os
from scipy.ndimage import zoom

def bin_data_fractional(data, binx, biny, binz):
    zoom_factors = (1 / binz, 1 / biny, 1 / binx)
    return zoom(data, zoom_factors, order=1)  # Linear interpolation

def get_mrc_voxel_size(mrc):
    voxel_size_x = mrc.header.cella.x / mrc.header.mx
    voxel_size_y = mrc.header.cella.y / mrc.header.my
    voxel_size_z = mrc.header.cella.z / mrc.header.mz
    return voxel_size_x, voxel_size_y, voxel_size_z

def mrc_to_hdf(input_file, output_file, binx, biny, binz, make_isotropic):
    with mrcfile.open(input_file, mode='r') as mrc:
        data = mrc.data
        voxel_size_x, voxel_size_y, voxel_size_z = get_mrc_voxel_size(mrc)
        if make_isotropic:
            binx, biny, binz = compute_isotropic_bins(voxel_size_x, voxel_size_y, voxel_size_z)
        binned_data = bin_data_fractional(data, binx, biny, binz)
        with h5py.File(output_file, 'w') as out_hdf:
            out_hdf.create_dataset('dataset', data=binned_data)
            out_hdf.attrs['apix_x'] = voxel_size_x * binx
            out_hdf.attrs['apix_y'] = voxel_size_y * biny
            out_hdf.attrs['apix_z'] = voxel_size_z * binz

def hdf_to_mrc(input_file, output_file, binx, biny, binz, make_isotropic):
    with h5py.File(input_file, 'r') as hdf:
        dataset_name = list(hdf.keys())[0]
        data = hdf[dataset_name][:]
        apix_x, apix_y, apix_z = hdf.attrs.get('apix_x', 1.0), hdf.attrs.get('apix_y', 1.0), hdf.attrs.get('apix_z', 1.0)
        if make_isotropic:
            binx, biny, binz = compute_isotropic_bins(apix_x, apix_y, apix_z)
        binned_data = bin_data_fractional(data, binx, biny, binz)
        with mrcfile.new(output_file, overwrite=True) as out_mrc:
            out_mrc.set_data(binned_data.astype(np.float32))
            out_mrc.voxel_size.x = apix_x * binx
            out_mrc.voxel_size.y = apix_y * biny
            out_mrc.voxel_size.z = apix_z * binz

def compute_isotropic_bins(apix_x, apix_y, apix_z):
    max_apix = max(apix_x, apix_y, apix_z)
    binx, biny, binz = apix_x / max_apix, apix_y / max_apix, apix_z / max_apix
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

    if input_ext == '.mrc' and output_ext == '.hdf':
        mrc_to_hdf(args.input_file, args.output_file, args.binx, args.biny, args.binz, args.make_isotropic)
    elif input_ext in ['.hdf', '.h5'] and output_ext == '.mrc':
        hdf_to_mrc(args.input_file, args.output_file, args.binx, args.biny, args.binz, args.make_isotropic)
    elif input_ext == '.mrc' and output_ext == '.mrc':
        mrc_to_hdf(args.input_file, args.output_file, args.binx, args.biny, args.binz, args.make_isotropic)
    elif input_ext in ['.hdf', '.h5'] and output_ext in ['.hdf', '.h5']:
        hdf_to_mrc(args.input_file, args.output_file, args.binx, args.biny, args.binz, args.make_isotropic)
    else:
        print("Unsupported file format conversion. Supported conversions: MRC<->HDF.")

if __name__ == '__main__':
    main()
'''
