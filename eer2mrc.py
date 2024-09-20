import numpy as np
import mrcfile
import argparse
from EMAN2 import *
from EMAN2_utils import *
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

print_lock = threading.Lock()

def process_file(f, options, img_type):
    start_time = time.time()
    out_img_file = os.path.join(options.path, f.replace(".eer", "_reduced.mrc"))
    n_subframes = EMUtil.get_image_count(f)
    avg = EMData()

    out_mode = 'float'
    out_type = EMUtil.get_image_ext_type("mrc")  # default to mrc
    not_swap = True  # typically not needed for mrc or hdf

    frames_per_final = n_subframes // options.n_final
    remainder_frames = n_subframes % options.n_final

    hdr = EMData(f, 0, True)
    nx = hdr['nx']
    ny = hdr['ny']
    out3d_img = EMData(nx, ny, options.n_final)
    
    if options.apix != 1.0:
        out3d_img["apix_x"] = options.apix
        out3d_img["apix_y"] = options.apix
        out3d_img["apix_z"] = options.apix

    for j in range(0, options.n_final):
        with print_lock:
            print(f'[{time.strftime("%H:%M:%S")}] Processing frame {j+1}/{options.n_final} for file {f}')

        if j == options.n_final - 1:
            upper_limit = frames_per_final + remainder_frames
        else:
            upper_limit = frames_per_final

        for i in range(upper_limit):
            img_indx = j * frames_per_final + i
            d = EMData()
            d.read_image(f, img_indx, False, None, False, img_type)

            if options.verbose and i % 100 == 0:  # Reducing print frequency
                with print_lock:
                    print(f'[{time.strftime("%H:%M:%S")}] File: {f}, Frame {i+1}/{upper_limit}, img_indx={img_indx}')

            # Running average
            if i == 0:
                avg = d.copy()
            else:
                avg *= i
                avg += d
                avg /= (i + 1)

        out3d_img.insert_clip(avg, (0, 0, j))

        with print_lock:
            print(f"[{time.strftime('%H:%M:%S')}] Inserted avg frame {j} into output image for file {f}")

    out3d_img.write_image(out_img_file, 0)
    duration = time.time() - start_time

    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] Saved averaged frames to {out_img_file}. Time taken: {duration:.2f} seconds for file {f}")

    return f

def main():
    parser = argparse.ArgumentParser(description="Average MRC frames and save to a new MRC file.")
    parser.add_argument("--apix", type=float, default=1.0, help="Set the sampling size in the output images if 1.0 is not correct.")
    parser.add_argument("--input_stem", help="String in input files; note that input EER files must end with .eer extension, so --input_stem=.eer might be good.")
    parser.add_argument("--outmode", type=str, default="float", help="Specify an alternate format (float, int8, int16, etc.).")
    parser.add_argument("--path", default='mrc_frames', help="Directory to store converted files. Default=mrc_frames_00")
    parser.add_argument("--n_final", type=int, default=10, help="Number of frames to have in the new mrc or hdf stack.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use for parallel processing.")

    eer_input_group = parser.add_mutually_exclusive_group()
    eer_input_group.add_argument("--eer2x", action="store_true", help="Render EER file on 8k grid.")
    eer_input_group.add_argument("--eer4x", action="store_true", help="Render EER file on 16k grid.")

    options = parser.parse_args()

    if options.n_final <= 0:
        print("Error: --n_final must be greater than 0.")
        sys.exit(1)

    img_type = IMAGE_UNKNOWN
    if options.eer2x:
        img_type = IMAGE_EER2X
    elif options.eer4x:
        img_type = IMAGE_EER4X

    files = [f for f in os.listdir('.') if options.input_stem in f and os.path.splitext(f)[-1].lower() == ".eer"]
    
    with print_lock:
        print(f'There are {len(files)} files to process: {files}')

    options = makepath(options, 'eer_frames_avgs')

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=options.threads) as executor:
        futures = [executor.submit(process_file, f, options, img_type) for f in files]
        for future in futures:
            try:
                result = future.result()
                with print_lock:
                    print(f'Processing completed for file: {result}')
            except Exception as e:
                with print_lock:
                    print(f'Error occurred: {e}')

if __name__ == "__main__":
    main()












'''
import numpy as np
import mrcfile
import argparse
from EMAN2 import *
from EMAN2_utils import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file(f, options, img_type):
    out_img_file = os.path.join(options.path, f.replace(".eer", "_reduced.mrc"))
    n_subframes = EMUtil.get_image_count(f)
    avg = EMData()
    
    out_mode = 'float'
    out_type = EMUtil.get_image_ext_type("mrc")  # default to mrc
    not_swap = True  # typically not needed for mrc or hdf

    frames_per_final = n_subframes // options.n_final  # how many input frames per final frame
    remainder_frames = n_subframes % options.n_final   # remainder frames to handle at the end

    hdr = EMData(f, 0, True)
    nx = hdr['nx']
    ny = hdr['ny']
    out3d_img = EMData(nx, ny, options.n_final)
    
    if options.apix != 1.0:
        out3d_img["apix_x"] = options.apix
        out3d_img["apix_y"] = options.apix
        out3d_img["apix_z"] = options.apix

    for j in range(0, options.n_final):
        if options.verbose:
            print(f'\nWorking on frame j={j+1}/{options.n_final} for file {f}')

        if j == options.n_final - 1:
            upper_limit = frames_per_final + remainder_frames
        else:
            upper_limit = frames_per_final

        if options.verbose:
            print(f'Averaging {upper_limit} frames for final frame {j+1} for file {f}')

        for i in range(upper_limit):
            img_indx = j * frames_per_final + i
            d = EMData()
            d.read_image(f, img_indx, False, None, False, img_type)
            if options.verbose:
                print(f'Read image i={i+1}/{upper_limit} for final frame {j+1}, img_indx={img_indx} for file {f}')

            # Running average
            if i == 0:
                avg = d.copy()
            else:
                avg *= i
                avg += d
                avg /= (i + 1)
        
        out3d_img.insert_clip(avg, (0, 0, j))
        if options.verbose:
            print(f"Inserted avg frame {j} into output image for file {f}")

    out3d_img.write_image(out_img_file, 0)
    if options.verbose:
        print(f"Saved averaged frames to {out_img_file}")
    
    return f

def main():
    parser = argparse.ArgumentParser(description="Average MRC frames and save to a new MRC file.")
    parser.add_argument("--apix", type=float, default=1.0, help="Default=1.0. Set the sampling size in the output images if 1.0 is not correct.")
    parser.add_argument("--input_stem", help="String in input files; note that input EER files must end with .eer extension, so --input_stem=.eer might be good.")
    parser.add_argument("--outmode", type=str, default="float", help="All EMAN2 programs write images with 4-byte floating point values by default. You can specify an alternate format (float, int8, int16, etc.).")
    parser.add_argument("--path", default='mrc_frames', help="Directory to store converted files. Default=mrc_frames_00")
    parser.add_argument("--n_final", type=int, default=10, help="Number of frames to have in the new mrc or hdf stack; e.g., if EER frames are 100, and --n_final is 10, then each 10 EER frames will be averaged together.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use for parallel processing. Default=4.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
   
    eer_input_group = parser.add_mutually_exclusive_group()
    eer_input_group.add_argument("--eer2x", action="store_true", help="Render EER file on 8k grid.")
    eer_input_group.add_argument("--eer4x", action="store_true", help="Render EER file on 16k grid.")

    options = parser.parse_args()

    if options.n_final <= 0:
        print("Error: --n_final must be greater than 0.")
        sys.exit(1)

    img_type = IMAGE_UNKNOWN
    if options.eer2x:
        img_type = IMAGE_EER2X
    elif options.eer4x:
        img_type = IMAGE_EER4X

    # Get all files matching the input stem
    files = [f for f in os.listdir('.') if options.input_stem in f and os.path.splitext(f)[-1].lower() == ".eer"]
    
    if options.verbose:
        print(f'there are {len(files)} files\nwhich are files={files}')

    # Make the output path
    options = makepath(options, 'eer_frames_avgs')

    # Use ThreadPoolExecutor to parallelize the processing
    with ThreadPoolExecutor(max_workers=options.threads) as executor:
        futures = {executor.submit(process_file, f, options, img_type): f for f in files}

        for future in as_completed(futures):
            f = futures[future]
            try:
                result = future.result()
                if options.verbose:
                    print(f"Completed processing for file: {result}")
            except Exception as exc:
                print(f"File {f} generated an exception: {exc}")

if __name__ == "__main__":
    main()
'''












'''
OLD 9/20/24

import numpy as np
import mrcfile
import argparse
from EMAN2 import *
from EMAN2_utils import *
import os

def main():
    parser = argparse.ArgumentParser(description="Average MRC frames and save to a new MRC file.")
    parser.add_argument("--apix", type=float, default=1.0, help="Default=1.0. Set the sampling size in the output images if 1.0 is not correct.")
    parser.add_argument("--input_stem", help="String in input files; note that input EER files must end with .eer extension, so --input_stem=.eer might be good.")
    parser.add_argument("--outmode", type=str, default="float", help="All EMAN2 programs write images with 4-byte floating point values by default. You can specify an alternate format (float, int8, int16, etc.).")
    parser.add_argument("--path", default='mrc_frames', help="Directory to store converted files. Default=mrc_frames_00")
    parser.add_argument("--n_final", type=int, default=10, help="Number of frames to have in the new mrc or hdf stack; e.g., if EER frames are 100, and --n_final is 10, then each 10 EER frames will be averaged together. Remainder frames will be averaged into the last new frame.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
   
    eer_input_group = parser.add_mutually_exclusive_group()
    eer_input_group.add_argument("--eer2x", action="store_true", help="Render EER file on 8k grid.")
    eer_input_group.add_argument("--eer4x", action="store_true", help="Render EER file on 16k grid.")

    options = parser.parse_args()

    if options.n_final <= 0:
        print("Error: --n_final must be greater than 0.")
        sys.exit(1)

    img_type = IMAGE_UNKNOWN
    if options.eer2x:
        img_type = IMAGE_EER2X
    elif options.eer4x:
        img_type = IMAGE_EER4X

    files = [f for f in os.listdir('.') if options.input_stem in f and os.path.splitext(f)[-1].lower() == ".eer"]
    if options.verbose:
        print(f'there are these many files n={len(files)}\nwhich are files={files}')

    options = makepath(options,'eer_frames_avgs')

    for f in files:
        if os.path.splitext(f)[-1].lower() != ".eer":
            print(f"Error: This program only works for EER files. The extension of file={f} is {os.path.splitext(f)[-1]}")
            sys.exit(1)

        out_img_file = os.path.join(options.path, f.replace(".eer", "_reduced.mrc"))

        #Read EER file
        n_subframes = EMUtil.get_image_count(f)
        avg = EMData()
        
        out_mode = 'float'
        out_type = EMUtil.get_image_ext_type("mrc")  # default to mrc
        not_swap = True  # typically not needed for mrc or hdf

        # Determine the number of frames to average per final frame
        frames_per_final = n_subframes // options.n_final  # how many input frames per final frame
        remainder_frames = n_subframes % options.n_final   # remainder frames to handle at the end

        hdr = EMData(f,0,True)
        nx = hdr['nx']
        ny = hdr['ny']
        out3d_img = EMData( nx, ny, options.n_final)
        if options.apix != 1.0:
            out3d_img["apix_x"] = options.apix
            out3d_img["apix_y"] = options.apix
            out3d_img["apix_z"] = options.apix
        

        for j in range(0, options.n_final):
            if options.verbose:
                print(f'\nWorking on frame j={j+1}/{options.n_final}')

            # Adjust number of frames in the last iteration if there's a remainder
            # (For the last frame, average both the regular number of frames and any remaining frames from the division ensuring  all  frames are used).
            if j == options.n_final - 1:
                upper_limit = frames_per_final + remainder_frames
            else:
                upper_limit = frames_per_final

            if options.verbose:
                print(f'Averaging {upper_limit} frames for final frame {j+1}')

            for i in range(upper_limit):
                img_indx = j * frames_per_final + i  # Correctly calculate the frame index
                d = EMData()
                d.read_image(f, img_indx, False, None, False, img_type)
                if options.verbose:
                    print(f'Read image i={i+1}/{upper_limit} for final frame {j+1}, img_indx={img_indx}')

                #RUNNING AVERAGE
                if i == 0:
                    avg = d.copy()
                else:
                    avg *= i  # Scale by the number of images averaged so far
                    avg += d
                    avg /= (i + 1)
            
            try:

                out3d_img.insert_clip(avg, (0, 0, j))
                print(f"Inserted avg frame {j} into output image")
                    
            except Exception as e:
                print(f"Error encountered while inserting frame {j}: {e}")
                raise

        print(f'inserted all frames; attempting to write output_type={type(out3d_img)} to file={out_img_file} with min={out3d_img["minimum"]},max{out3d_img["maximum"]},sigma={out3d_img["sigma"]}')
        out3d_img.write_image(out_img_file, 0)
        if options.verbose:
            print(f"Saved averaged frames to {out_img_file}")
        
        print('\nDONE')

if __name__ == "__main__":
    main()
'''
