#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya, 03/2023; last update 03/2023

import os
import argparse
from PIL import Image
from EMAN2 import *
from EMAN2_Utils import *

def main():
    parser = argparse.ArgumentParser(description="""Create a gallery of 2D cryoEM images in the current directory or from an input stack with 
        multiple 2D images and prints them to a PDF file.""")

    parser.add_argument("--input_file", type=str, required=True, default=None, help="Default=None. Path to 2D image stack")
    parser.add_argument("--input_string", type=str, required=False, default=None, help="Default=None. String that should be contained in files to process")
    parser.add_argument("--output_file", type=str, required=True, default=None, help="Default=NOne. Name of PDF file to save image gallery")

    args = parser.parse_args()

    files=[]
    images=[]
    if args.input_file:
        n=EMUtil.get_image_count(args.input_file)
        print(f"\nstack={args.input_file} has these many images in it n={n}")
        files.append(args.input_file)
    elif args.input_string:
        files = [f for f in os.listdir(os.getcwd()) if os.path.splitext()[-1] in extensions()]

    for f in files:
        n=EMUtil.get_image_count(args.input_file)
        print(f"\nprocessing file {f} with has these many images in it n={n}")

        for i in range(n):
            image = EMData(args.input_file,i)
            #image.process_inplace("threshold.clampminmax.nsigma",{"nsigma":3})
            # convert image to RGB mode
            image = image.numpy().astype('uint8')
            image = Image.fromarray(image, mode='L')
            image = image.convert('RGB')
            images.append(image)

    images[0].save(args.output_file, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:])


if __name__ == '__main__':
    main()
