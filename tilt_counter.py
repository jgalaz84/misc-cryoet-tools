#!/usr/bin/env python
#
# Author: Jesus G. Galaz-Montoya, 06/05/2012 - Modified 02/Dec/2024

import os
import random
import string
from EMAN2_utils import *
import time
from datetime import timedelta

def main():
	start = time.perf_counter()
	progname = os.path.basename(sys.argv[0])
	
	usage = """Count images in all tiltseries in a given directory and produce an output .txt file"""
		
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)

	parser.add_argument("--idindx", type=int, default=None, help="""Default=None. Where in the filename is the 'ID' of the tomogram. 
		For example, tomo_01_whatever.hdf would have it in index '1', counting strng elements divided by '_', and starting with index 0.""")
	
	parser.add_argument("--n_expected", type=int, default=None, help="""Default=None. Number of tilt images expected for each tilt series 
		(can be calculated as full-tilt-range/tilt-step + 1. E.g., for a tilt series from -60 to +60 deg, the tilt range is 120; 
		if the tilt step is 1 deg, one expects to have 121 images.""")

	parser.add_argument("--ppid", type=int, default=-1, help="Default=-1. Set the PID of the parent process, used for cross platform PPID")

	parser.add_argument("--tag", type=str, default='', help="""Default=None. String to add to the output filename. Output default is iamge_count.txt; 
		if you supply tag=tilt, then output will be tilt_image_count.txt""")
	parser.add_argument("--targetdir", type=str, default='.', help="""Default=current directory. Path to the directory containing the tiltseries 
		whose number of images will be counted.""")
	parser.add_argument("--threshold_to_exclude", type=float, default=None, help="""Default=None. Requires --n_expected. 
		Fraction (from 0.0 to 1.0) of --n_expected required to be present to keep the tilt series for downstream analyses. 
		E.g., in a tilt series with --n_expected=121, if --threshold_to_exclude=0.5, tilt series will fewer than 61 images 
		will be compartmentalized in a subfolder "z.bad".""")

	(options, args) = parser.parse_args()


	logger = E2init(sys.argv, options.ppid)

	ts = [ options.targetdir + '/' + t for t in os.listdir(options.targetdir) if '.tlt' not in t[-5:] and '.rawtlt' not in t[-7:] and '.txt' not in t[-4:] and not os.path.isdir(t)]

	#extensions = extensions()
	extensions = ['.dm3', '.DM3', '.mrc', '.MRC', '.mrcs', '.MRCS', '.hdf', '.HDF', '.tif', '.TIF', '.st', '.ST', '.ali', '.ALI', '.rec', '.REC']

	#clean up the potential tilt series files
	for t in ts:
		filename, file_ext = os.path.splitext(t)
		if file_ext not in extensions:
			ts.remove(t)
		try:
			hdr=EMData(t,0,True)
		except:
			if t in ts:
				ts.remove(t)
				print("\nskipping file {} because it doesn't seem to be a readable tilt series or has too few images".format(t))

	ts.sort()
	n = len(ts)
	print('\nfound n={} files'.format(n))

	lines=[]

	for t in ts:
		ID = os.path.basename(t).split('_')[options.idindx]
		
		hdr = EMData(t,0,True)
		nz = hdr['nz']

		filename, file_ext = os.path.splitext(t)
		
		line = ID + '\t' + str(nz) + '\n'
		lines.append(line)

		if options.threshold_to_exclude:
			if options.n_expected:
				if float(nz)/float(options.n_expected) < options.threshold_to_exclude:
					fsindir = os.listdir(os.getcwd())
					if "z.bad" not in fsindir:
						bad_dir="z.bad"
						if options.targetdir:
							bad_dir = options.targetdir +'/z.bad'
						os.mkdir(bad_dir)
					os.rename(t,bad_dir+'/'+t)
					try:
						os.rename(t.replace(file_ext,'.rawtlt'),bad_dir+'/'+t.replace(file_ext,'.rawtlt'))
					print("\nWARNING: moved file f={} to {} because {}/{}={} was lower than {}".format(t,bad_dir,nz,options.n_expected,round(float(nz)/float(options.n_expected),4),options.threshold_to_exclude))

			else:
				print("\nERROR: --threshold_to_exclude requires --n_expected. EXITING")
				sys.exit(1)

	letters = string.ascii_lowercase

	tag = ''.join( random.choice(letters) for i in range(6))
	if options.tag:
		tag = options.tag
	
	outputfile = tag + '_img_count.txt'
	with open( outputfile, 'w' ) as f:
		f.writelines(lines)

	print('\nwrote image count to {}'.format(outputfile))
	elapsed = time.perf_counter() - start
	print(str(timedelta(seconds=elapsed)))
	
	E2end(logger)
	return


if '__main__' == __name__:
	main()