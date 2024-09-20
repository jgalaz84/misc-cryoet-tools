import numpy as np
import mrcfile
import argparse
from EMAN2 import *
from EMAN2_utils import *
import os

def main():
    parser = argparse.ArgumentParser(description="Average MRC frames and save to a new MRC file.")
    parser.add_argument("--apix", type=float, default=1.0, help="Default=1.0. Set the sampling size in the output images if 1.0 is not correct.")
    #parser.add_argument("--compressbits", type=int, help="HDF only. Bits to keep for compression. -1 for no compression", default=-1)
    parser.add_argument("--input_stem", help="String in input files; note that input EER files must end with .eer extension, so --input_stem=.eer might be good.")
    parser.add_argument("--outmode", type=str, default="float", help="All EMAN2 programs write images with 4-byte floating point values by default. You can specify an alternate format (float, int8, int16, etc.).")
    #parser.add_argument("--output", default=None, help="Default=None. Output MRC file")
    parser.add_argument("--path", default='mrc_frames', help="Directory to store converted files. Default=mrc_frames_00")
    parser.add_argument("--n_final", type=int, default=10, help="Number of frames to have in the new mrc or hdf stack; e.g., if EER frames are 100, and --n_final is 10, then each 10 EER frames will be averaged together. Remainder frames will be averaged into the last new frame.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
   
    eer_input_group = parser.add_mutually_exclusive_group()
    eer_input_group.add_argument("--eer2x", action="store_true", help="Render EER file on 8k grid.")
    eer_input_group.add_argument("--eer4x", action="store_true", help="Render EER file on 16k grid.")

    options = parser.parse_args()

    #if options.compressbits >= 0 and not options.output.endswith(".hdf"):
    #    print(f'\nERROR: --compressbits requires .hdf output. Current file has extension={os.path.splitext(options.output)[-1]}')
    #    sys.exit(1)

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

    for f in files:
        if os.path.splitext(f)[-1].lower() != ".eer":
            print(f"Error: This program only works for EER files. The extension of file={f} is {os.path.splitext(f)[-1]}")
            sys.exit(1)

        out_img_file = os.path.join(options.path, f.replace(".eer", "_reduced.mrc"))

        #Read EER file
        #n_eer = EMUtil.get_image_count(options.input)
        n_subframes = EMUtil.get_image_count(f)
        avg = EMData()
        
        out_mode = 'float'
        out_type = EMUtil.get_image_ext_type("mrc")  # default to mrc
        not_swap = True  # typically not needed for mrc or hdf

        #if options.output.endswith(".mrc"):
        #    out_mode = file_mode_map["float"]  # or adjust to int16, int8 if needed
        #elif options.output.endswith(".hdf"):
        #    out_type = EMUtil.get_image_ext_type("hdf")
        #    out_mode = file_mode_map["float"]  # or another format if needed
        #else:
        #    raise ValueError("Unsupported output format. Only .mrc and .hdf are supported.")

        options = makepath(options,'sptsim')

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

                #CUMMULATIVE AVERAGING
                #if i == 0:
                #    avg = d.copy()
                #else:
                #    avg += d
                #avg /= upper_limit  # Final averaging after accumulating all frames

            
            try:
                #if options.compressbits >= 0 and out_type == EMUtil.get_image_ext_type("hdf"):
                #    avg.write_compressed( os.path.join(options.path, out_img_file), j, options.compressbits, nooutliers=True)
                #else:
                    #avg.write_image(os.path.join(options.path, out_img_file), j, out_type, False, None, out_mode, not_swap)
                #region = Region(0, 0, j, nx, ny, 1)
                #avg.write_image(os.path.join(options.path, out_img_file), 0, out_type, False, region, out_mode, not_swap)
                out3d_img.insert_clip(avg, (0, 0, j))
                print(f"Inserted avg frame {j} into output image")
                    
            except Exception as e:
                print(f"Error encountered while writing frame {j}: {e}")
                raise

        out3d_img.write_image(out_img_file, 0)
        if options.verbose:
            print(f"Saved averaged frames to {out_img_file}")
        
        print('\nDONE')

if __name__ == "__main__":
    main()



'''
ORIG
def main():
    parser = argparse.ArgumentParser(description="Average MRC frames and save to a new MRC file.")
    parser.add_argument("--compressbits", type=int,help="HDF only. Bits to keep for compression. -1 for no compression",default=-1)
    #parser.add_argument("--exclude_n", default=0, help="Default=0. Number of frames to exclude from the beginning of the eer stack.")
    parser.add_argument("--input", help="Input MRC file")
    parser.add_argument("--outmode", type=str, default="float", help="All EMAN2 programs write images with 4-byte floating point values when possible by default. This allows specifying an alternate format when supported (float, int8, int16, int32, uint8, uint16, uint32). Values are rescaled to fill MIN-MAX range.")
    parser.add_argument("--output", default=None, help="Default=None. Output MRC file")
    parser.add_argument("--path", default='mrc_frames', help="Numbered directory where to store converted files. Default=mrc_frames_00")
    parser.add_argument("--n_final", type=int, default=10, help="Number of frames to have in the new mrc or hdf stack; for example, if eer frames are 100, and --n_final is 10, then each 10 eer frames will be averaged together. Remainder frames will be averaged into the last new frame.")
    
    eer_input_group = parser.add_mutually_exclusive_group()
    eer_input_group.add_argument("--eer2x", action="store_true", help="Render EER file on 8k grid.")
    eer_input_group.add_argument("--eer4x", action="store_true", help="Render EER file on 16k grid.")

    options = parser.parse_args()

    if options.input[-4:] != ".eer":
        print(f"Error: This program only works for EER files. The extension of --input is {os.path.splitext(options.input)[-1]}")
        sys.exit(1)

    if options.compressbits>=0 and not options.output.endswith(".hdf"):
        print(f'\nERROR: --compressbits requires .hdf output. Current file has extension={os.path.splitext(options.output)[-1]}')
        sys.exit(1)

    img_type = IMAGE_UNKNOWN 
    if options.eer2x:
        img_type = IMAGE_EER2X
    elif options.eer4x:
        img_type = IMAGE_EER4X

    # Read EER file
    n_eer=EMUtil.get_image_count(options.input)
    
    avg = EMData()
    
    out_mode = 'float'
    out_type = EMUtil.get_image_ext_type("mrc") #default to mrc
    not_swap = True #presumably typically not needed for mrc or hdf

    if options.output is None:
        options.output = options.input.replace(".eer", "_reduced.mrc")

    if options.output.endswith(".mrc"):
        out_mode = file_mode_map["float"]  # or adjust to int16, int8 if needed
    elif options.output.endswith(".hdf"):
        out_type = EMUtil.get_image_ext_type("hdf")
        out_mode = file_mode_map["float"]  # or another format if needed
    else:
        raise ValueError("Unsupported output format. Only .mrc and .hdf are supported.")

    upper_limit = n_eer//n_final
    remainder_frames = n_eer%n_final
    for j in range(0, n_final):
        print(f'\nworking on frame j={j+1}/{n_final}')
        
        if j == n_final - 1: #this is the last iteration; remainder images need to go into the last average
            upper_limit = n_final + remainder_frames
        print(f'which will be an average of n={upper_limit} frames')

        for i in range(0,upper_limit):
            img_indx = (i+1)*(j+1) - 1
            d = EMData()
            d.read_image(options.input, img_indx, False, None, False, img_type)
            print(f'read image i={i} for frame j={j} which corresponds to img_indx={img_indx}\n')
            if i == 0: 
                avg = d.copy()
            elif i > 0:
                avg *= i #scale by the number of images averaged so far
                avg += d
                avg/=(i+1)
   
        if options.compressbits>=0 and out_type == EMUtil.get_image_ext_type("hdf"):
            #avg.write_compressed(options.output,-1, options.compressbits,nooutliers=True) #only supported for hdf
            avg.write_compressed(options.output,j, options.compressbits,nooutliers=True)
        else:
            #avg.write_image(options.output, -1, out_type, False, None, out_mode, not_swap)
            avg.write_image(options.output, j, out_type, False, None, out_mode, not_swap)
    
        print(f"Saved averaged frames to {options.output} index j={j}")
    print('\nDONE')

if __name__ == "__main__":
    main()
'''