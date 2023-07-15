#!/usr/bin/env python
# Author: Jesus Galaz-Montoya 06/2023; last modification: 07/2023

import os
import shutil
from EMAN2 import *
from EMAN2_utils import *
import time
from datetime import timedelta


def main():
	print("\n(ofsc.py)(main)")
	start = time.perf_counter()

	progname = os.path.basename(sys.argv[0])
	usage = """prog img1 img2 [options]
	This programs calculates the average "directional" FSC between two 3D images (provided as arguments at the command line) 
	by averaging the 2D FSCs between corresponding slices along the x, y and z axis directions.
	"""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)
	parser.add_argument("--goldstandard", action="store_true", default=False, help="""Default=False. Supply this if computing ofsc between even and odd reconstructions.""")
	
	parser.add_argument("--path", type=str, default='ofsc',help="""Defautl=ofsc. Directory to store results in. The default is a numbered series of directories 
		containing the prefix 'ofsc'; for example, ofsc_02 will be the directory by default if 'ofsc_01' already exists.""")

	parser.add_argument("--removetmp", action="store_true", default=False, help="""Default=False. Remove the unstacked directional slices for both input volumes.""")

	parser.add_argument("--test", action="store_true", default=False, help="""Default=False. Test run looking at one axis only.""")

	parser.add_argument("--verbose", "-v", type=int, default=0, help="Default 0. Verbose level [0-9], higner number means higher level of verboseness.")

	(options, args) = parser.parse_args()
	
	img1 = sys.argv[1]
	img2 = sys.argv[2]

	tag1 = '1'
	tag2 = '2'
	if options.goldstandard:
		tag1 = 'even'
		tag2 = 'odd'

	makepath(options,stem='ofsc')
	
	#logger = E2init(sys.argv, options.ppid)
	logger = E2init(sys.argv)

	fscs_dict = calc_slices_fscs(options,img1,img2,tag1,tag2)

	calc_ofscs(options, fscs_dict)

	E2end(logger)

	elapsed = time.perf_counter() - start	
	print(str(timedelta(seconds=elapsed)))

	return


def calc_slices_fscs(options, img1, img2, tag1='', tag2=''):
	print("\n(ofsc.py)(calc_slices_fscs)")

	fsc_file = options.path +' /' + os.path.splitext( os.path.basename(img1) )[0] + '_fsc.txt'
	cmd_fsc = 'e2proc3d.py ' + img1 + ' ' + fsc_file + ' --calcfsc ' + img2
	print(f"\n(ofsc)(calc_slices_fscs) cmd_fsc={cmd_fsc}")
	runcmd(options,cmd_fsc)

	#print(f"\n\n\n\n!!!!!!!!!11111\n(ofsc.py)(calc_slices_fscs) tag1={tag1}")
	slices_dir1, slices_dict1 = unstack_img(options, img1, tag1)
	
	#print(f"\n\n\n\n!!!!!!!!!22222\n(ofsc.py)(calc_slices_fscs) tag2={tag2}")
	slices_dir2, slices_dict2 = unstack_img(options, img2, tag2)
	
	fscs_dict = {}
	i=0
	for axis in slices_dict1:
		if options.test and i > 0:
			break
		if options.verbose:
			print(f"\n(ofsc)(calc_slices_fscs) calling fsc function for axis={axis}")
		fsc_files = slices_fscs(options,slices_dict1[axis],slices_dict2[axis],axis)
		fscs_dict.update({axis:fsc_files})
		i+=1

	if options.removetmp:
		shutil.rmtree(slices_dir1, ignore_errors=True)
		shutil.rmtree(slices_dir2, ignore_errors=True)
	
	return fscs_dict


def unstack_img(options, img, tag):
	if options.verbose:
		print(f"\n(ofsc.py)(unstack_img); received tag={tag}")

	ext = os.path.splitext(img)[-1]
	
	cmd_slices = 'e2slicer.py ' + img + ' --path ' + options.path + '/slices_'+ tag +' --allx'
	if not options.test:
		cmd_slices +=' --ally --allz'

	runcmd(options,cmd_slices)

	slices_dir = os.getcwd()+ '/' + options.path + '/slices_01/'
	if tag:
		slices_dir = os.getcwd()+ '/'+ options.path + '/slices_' + tag + '_01/'
		if tag.isdigit():
			slices_dir_renamed = os.getcwd() + '/' + options.path + '/slices_0' + tag + '/'
			os.rename(slices_dir,slices_dir_renamed)
			slices_dir = slices_dir_renamed

	slices_dict = {}
	i=0
	for axis in ['x','y','z']:
		if options.test and i > 0:
			break

		slices_f = slices_dir + os.path.basename(img).replace(ext,'_all_' + axis + '.hdf')
		cmd_slices_unstack = 'e2proc2d.py ' + slices_f + ' ' + slices_f.replace('.hdf','_unstacked.hdf') + ' --unstacking --writejunk'
		print(f'\n{axis} unstacking cmd={cmd_slices_unstack}')
		runcmd(options,cmd_slices_unstack)
		fs = [slices_dir + f for f in os.listdir(slices_dir) if 'all_'+ axis + '_unstacked-' in f]
		fs.sort()
		slices_dict.update({axis:fs})

		i+=1

	return slices_dir, slices_dict


def slices_fscs(options,images_even,images_odd,axis=''):
	if options.verbose:
		print("\n(ofsc.py)(slices_fscs)")
	
	if options.verbose > 8:
		print(f"\nimages_even={images_even}")
	
	ext = os.path.splitext(images_even[0])[-1]

	fscs_dir = os.getcwd() + '/' + options.path +'/fscs_' + axis
	os.mkdir(fscs_dir)

	cmds_slices_fscs = ['e2proc3d.py ' + images_even[i] + ' ' 
	+ fscs_dir +'/fsc_' + axis + '_' + os.path.basename(images_even[i]).split('unstacked-')[-1].replace(ext,'.txt') + 
	' --calcfsc ' + images_odd[i] for i in range(len(images_even))]

	i=1
	n=len(cmds_slices_fscs)
	for cmd in cmds_slices_fscs:
		if options.verbose:
			print(f"\nCalculating FSC {i}/{n}, with cmd={cmd}")
		runcmd(options,cmd)
		i+=1
	
	fsc_files = [fscs_dir + '/fsc_' + axis + '_' + os.path.basename(images_even[i]).split('unstacked-')[-1].replace(ext,'.txt') for i in range(len(images_even))]

	return fsc_files


def calc_ofscs(options,fscs_dict):
	axis_areas_dict={}
	i=0
	for axis in fscs_dict:
		if options.test and i > 0:
			break
		
		fsc_files = fscs_dict[axis]
		vals_dict={}
		for f in fsc_files:
			num = f.split('_')[-1].replace('.txt','')
			print(f"\n(ofsc.py)(calc_ofscs) looking at file {f}")
			with open(f,'r') as g:
				lines=g.readlines()
				vals=[ float( line.replace('\n','').replace('\t',' ').split()[-1] ) for line in lines ]
				area = sum(vals)
				print(f"\n(ofsc.py)(calc_ofscs) axis={axis}, num={num}, area={area}")
				vals_dict.update({num:area})

		fsc_areas_f = os.getcwd() + '/' + options.path + '/ofsc_areas_' + axis + '.txt'
		with open(fsc_areas_f,'w') as h:
			lines = [str(num)+'\t'+str(vals_dict[num])+'\n' for num in vals_dict]
			#print(f"\nfsc_areas_f lines={lines}")
			h.writelines(lines)

		axis_area = sum(vals_dict.values())
		axis_areas_dict.update({axis:axis_area})
		i+=1

	print(f"\naxis_areas_dict={axis_areas_dict}")

	axis_areas_f = os.getcwd() + '/' + options.path + '/ofsc_areas_totals.txt'
	with open(axis_areas_f,'w') as j:
		lines = [axis+'\t'+str(axis_areas_dict[axis])+'\n' for axis in ['x','y','z']]
		j.writelines(lines)

	return axis_areas_dict


if __name__ == "__main__":
    main()
    sys.stdout.flush()