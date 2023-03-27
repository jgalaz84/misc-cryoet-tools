#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 11/2022; last modification: 03/2023
#
# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA

from __future__ import print_function
from __future__ import division
from past.utils import old_div

import numpy
import re

import os
import sys

from EMAN2 import *
from EMAN2_utils import *

import time
from datetime import timedelta


def main():
	start = time.perf_counter()

	progname = os.path.basename(sys.argv[0])
	usage = """prog [options]
	Program to genrate masks around objects in tomograms as a first level of coarse segmentation, to get rid of unnecessary background that may confuse segmentation and/or
	particle picking/template matching algorithms or simply yield an unnecessarily high number of false positives.
	Requires a coordinates file in .txt format with at least 4 columns: x y z d, where d is the diameter of the particle centered at x y z coordinates.
	Alternatively, if only 3 columns are present for the coordinates, a --diameter value is required. 
	"""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)

	parser.add_argument("--apix",type=float,default=None,help="""Default=None. Sampling size of the --input tomogram.""")

	parser.add_argument("--coords", type=str, default=None, help="""Default=None. Path to tomogram to apply mask to.""")
	
	parser.add_argument("--diameter", type=int, default=None, help="""Default=None. Diameter value in pixels, required if all particles are homogenous in size and coords file only contains x y z column values.""")

	parser.add_argument("--expand_mask", type=int, default=4, help="""Default=4. Number of pixels to expand mask by so that it is not tight on the structures being masked.""")
	parser.add_argument("--extract_ptcls", action="store_true", default=False, help="""Default=False. In addition to parsing coords, save particles from the tomogram in a stack.""")

	parser.add_argument("--id", type=str, default=None, help="""Default=None. Tag unique to the tomogram being processed; for example 't07'.""")
	parser.add_argument("--input", type=str, default=None, help="""Default=None. Path to tomogram to apply mask to.""")

	parser.add_argument("--path", type=str, default='coord_masks',help="""Defautl=tomo_shapes. Directory to store results in. The default is a numbered series of directories containing the prefix 'tomo_shapes'; for example, tomo_shapes_02 will be the directory by default if 'tomo_shapes_01' already exists.""")

	parser.add_argument("--ppid", type=int, default=-1, help="""Default=-1. Set the PID of the parent process, used for cross platform PPID.""")
	parser.add_argument("--savemasks", action="store_true", default=False, help="""Default=False. Saves masks as a stack of single particles.""")

	parser.add_argument("--verbose", "-v", type=int, default=0, help="Default 0. Verbose level [0-9], higner number means higher level of verboseness.")

	(options, args) = parser.parse_args()

	#c:log each execution of the script to an invisible log file at .eman2log.txt
	logger = E2init(sys.argv, options.ppid)

	#c:check that the --input tomogram is sane by trying to read the header, else exit
	try:
		hdr = EMData(options.input,0,True)
	except:
		print("\nERROR reading image --input={}".format(options.input))
		sys.exit(1)

	#c:determine the apix value to work with
	hdr = EMData(options.input,0,True)
	apix = round(float(hdr['apix_x']),2)
	if options.apix:
		if round(float(options.apix),2) != apix:
			print("\nWARNING: --apix={} doesn't match the value in the header, hdr['apix_x']={}".format(hdr['apix_x']))
			apix = options.apix

	lines=[]
	#c:open the --coords file and read its contents
	with open(options.coords,'r') as f:
		lines = f.readlines()

	coords_dict = {}
	coords=[]
	
	#c:make a directory where to store the output and temporary files
	makepath(options,stem='coord_masks')

	#c:continue only if read lines are not empty and no errors occured
	if lines:
		if options.verbose>5:
			print("\nFile provided --coords={} contains n={} lines".format(options.coords,len(lines)))
		
		#c:eliminate header lines that contain letters
		clines = [line for line in lines if not any(c.isalpha() for c in line) ]

		indexes_to_remove = []
		for i in range(len(clines)):
			print(f"\nexamining cline {i} which is\n{clines[i]}\nto determine whether it needs to be removed")
			if clines[i].isalpha() or len(clines[i])<5:
				indexes_to_remove.append(i)	
		
		for j in indexes_to_remove:
			prit(f"\nremoving line index {j} which is {clines[j]}")
			clines.pop(j)

		print("\nlen(clines)={}".format(len(clines)))
		
		#c:lists to store minor and major axes lengths, as well as calculated volumes
		majors=[]
		minors=[]
		vols = []

		#c:if there are 7 values in each line, it means there are (aberrantly) 2 measurements for each minor and major axis, one in pixels one in angstroms, so wee need to analyze both measurements
		#c:lists to store minor and major axes lengths in pixels (second manual measurement)
		majorps=[]
		minorps=[]

		#c:lists to store the differences between major-majorp and minor-minorp to later estimate manual measurement error
		majords=[]
		minords=[]

		#c:lists to store the average of the repeat axis length measurements
		majoras=[]
		minoras=[]
		
		'''
		if optiosn.verbose:
			print("\nclines[0]={}".format(clines[0]))
			print("\nclines[0].split()={}".format(clines[0].split()))
			print("\nlen(clines[0])={}".format(len(clines[0].split())))
		'''

		#c:loop through the "clean lines" to parse them and populate the empty lists created above
		k=0
		for line in clines:
			print(f"\nexamining clean line {line}")
			lsplit=line.split()
			line_elements_n = len(lsplit)

			major = None 
			if line_elements_n > 4:
				major=round(float(lsplit[3]),2)
			else:
				try:
					major = options.diameter/1.0
				except:
					print("\nERROR: --diameter required if --coords file contains only x y z values.")
					sys.exit(1)

			majors.append(major)

			minor=None

			if line_elements_n > 4:
				minor=round(float(lsplit[4]),2)
				minors.append(minor)

			if line_elements_n > 5:
				majorp=round(float(lsplit[5]),2)
				majorps.append(majorp)

				minorp=round(float(lsplit[6]),2)
				minorps.append(minorp)

				majord=round(math.fabs(float(major) - float(majorp)*apix),2)
				majords.append(majord)

				minord=round(math.fabs(float(minor) -  float(minorp)*apix),2)
				minords.append(minord)

				majora=round((float(major) + float(majorp)*apix)/2.0,2)
				major = majora #c:if two measurements were present, replace major with the average of major and majorp
				majoras.append(majora)

				minora=round((float(minor) +  float(minorp)*apix)/2.0,2)
				minor = minora #c:if two measurements were present, replace minor with the average of minor and minorp
				minoras.append(minora)
			
			#c:because of the missing wedge, we only have major and minor axes presumably through the central slice of each object; we asssume the axis in z to be the same as 'minor'
			vol = None
			if minor:
				coords_dict.update( { k:{'x':lsplit[0], 'y':lsplit[1], 'z':lsplit[2], 'major':major, 'minor':minor }})
				vol = (4.0/3.0)*math.pi*( (major/2.0) * (minor/2.0) * (minor/2.0))
			else:
				coords_dict.update( { k:{'x':lsplit[0], 'y':lsplit[1], 'z':lsplit[2], 'major':major}})
				vol = (4.0/3.0)*math.pi*((major/2.0)**3)
			if vol:
				vols.append(vol)
			else:
				print("\nWARNING: could not calculate volume for particle at coorinates line={}".format(line))

			k+=1

		#c:some measurements need to be stored only if there are two measurements for each major and minor axes
		if len( clines[0].split() ) > 5:
			#c:if there were two manual measurements of major and minor axes, use their average as the final measurements
			majors = majoras.copy()
			minors = minoras.copy()

			majords_mean = round(numpy.mean(majords),2)
			majords_std = round(numpy.std(majords),2)
			
			minords_mean = round(numpy.mean(minords),2)
			minords_std = round(numpy.std(minords),2)

			if options.verbose > 3:
				print("\nMean differences and standard deviations in Angstroms are: majords_mean={}, majords_std={}, minords_mean={}, minords_std={}".format( majords_mean, majords_std, minords_mean, minords_std))

			majoras_mean = round(numpy.mean(majoras),2)
			majoras_std = round(numpy.std(majoras),2)	
			
			minoras_mean = round(numpy.mean(minoras),2)
			minoras_std = round(numpy.std(minoras),2)
			
			if options.verbose > 3:
				print("\nMean averages and standard deviations in Angstroms are: majoras_mean={}, majoras_std={}, minoras_mean={}, minoras_std={}".format( majoras_mean, majoras_std, minoras_mean, minoras_std))
				print("\nThose reflect the overall average major and minor axess across VLPs for this tomogram")

			output_majords_lines = [str(mad)+'\n' for mad in majords]
			with open(options.path+'/'+ options.id+'_major_diffs.txt','w') as c:
				c.writelines(output_majords_lines)

			output_minords_lines = [str(mid)+'\n' for mid in minords]
			with open(options.path+'/'+ options.id+'_minor_diffs.txt','w') as d:
				d.writelines(output_minords_lines)

		#c:these need to be stored regardless of whether there are single or double measurements for major and minor axes
		output_majors_lines = [str(ma)+'\n' for ma in majors]

		#c:if there's only one measurement because the particles are assumed to be spherical, use 'diameters' instead of 'majors' (for major axis) in the output filename
		output_f_majors = options.path+'/'+ options.id+'_diameters.txt'
		if len( clines[0].split() ) > 4:
			output_f_majors = options.path+'/'+ options.id+'_majors.txt'

			output_minors_lines = [str(mi)+'\n' for mi in minors]
			with open(options.path+'/'+ options.id+'_minors.txt','w') as b:
				b.writelines(output_minors_lines)

		with open(output_f_majors,'w') as a:
			a.writelines(output_majors_lines)


		output_vols_lines = [str(v)+'\n' for v in vols]
		with open(options.path+'/'+ options.id+'_vols.txt','w') as e:
			e.writelines(output_vols_lines)


		#c: after sucessful parsing and analysis of coords and sizes, produce the masked version of the tomogram
		if coords_dict:
			#c: make an empty volume of the same size of the tomogram and set the pixel values to zero
			masks_vol = EMData(hdr['nx'],hdr['ny'],hdr['nz'])
			masks_vol.to_zero()

			nptcls=len(coords_dict)
			
			#c: make a generous box (~1.75 the max span of the largest particle), multiple of 8 in case it helps with speed for other yet-to-determine processes
			boxsize = 8*round( (max(majors+minors)/apix) * 1.75 / 8)
			if options.verbose > 3:
				print("\nboxsize={}".format(boxsize))

			mask=EMData(boxsize,boxsize,boxsize)
			mask.to_one()
			for i in range(nptcls):
				if options.verbose > 3:
					print("\nworking on particle={}/{}".format(i,nptcls))
				
				x=int(coords_dict[i]['x'])
				y=int(coords_dict[i]['y'])
				z=int(coords_dict[i]['z'])
				radius=int(round((coords_dict[i]['major']/2)/apix))
				if options.verbose > 3:
					print("\nfor masking, radius is={} for ptcl {}".format(radius,i))

				if options.expand_mask:
					radius+=options.expand_mask
					if options.verbose > 3:
						print("\ngiven --expand_mask={}, radius+expansion={}".format(options.expand_mask,radius))


				#c:make spherical mask around each ptcl x,y,z using the ptcl's largest axis, expand it if requested, lowpass to soften, insert in output masks vol
				maski=mask.copy()

				maski.process_inplace("mask.sharp",{"outer_radius":radius})

				if options.savemasks:
					maski.write_image(options.path+'/'+options.id+'_masks_n'+str(nptcls).zfill(len( str(nptcls) ))+'.hdf',i)

				masks_vol.insert_scaled_sum(maski,[x,y,z])

				sys.stdout.flush()

				if options.extract_ptcls:
					#c:make the boxsize 1.5 times larger than the largest axis measured (as required for analysis), and round to the nearest multiple of 8 for future speed 

					r=Region(x-old_div(boxsize,2),y-old_div(boxsize,2),z-old_div(boxsize,2),boxsize,boxsize,boxsize)
					img=EMData(options.input,0,0,r)
					
					img.write_image(options.path+'/'+options.id+'_stack_n'+str(nptcls).zfill(len( str(nptcls) ))+'.hdf',i)

				if options.verbose > 3:
					print("\nfinished processing particle={}/{}".format(i,nptcls))

			#c:make the volume with all the masks "soft"
			halfnyquist=1/(2*(2*apix))
			if options.verbose > 3:
				print("\nhalfnyquist is at freq={}, or {} in Angstroms".format(halfnyquist,1.0/halfnyquist))
			masks_vol.process_inplace("filter.lowpass.gauss",{"cutoff_freq":halfnyquist})

			masks_vol.write_image(options.path+'/'+options.id+'_masks_tomo.hdf')
			tomo = EMData(options.input)
			tomo_masked = tomo*masks_vol
			tomo_masked.write_image(options.path+'/'+options.id+'_masked.hdf')

		else:
			print("\nWARNING: coords_dict seems empty, {}".format(coords_dict))
	else:
		print("\nERROR: file provided --coords={} seems to be empty".format(options.coords))

	E2end(logger)
	
	elapsed = time.perf_counter() - start
	
	print(str(timedelta(seconds=elapsed)))
	
	return


if __name__ == "__main__":
    main()
    sys.stdout.flush()