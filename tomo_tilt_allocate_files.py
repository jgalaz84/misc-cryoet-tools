#!/usr/bin/env python
#
# Author: Jesus Galaz, 2018? - Last change 9/nov/2023
# Copyright (c) 2011 Baylor College of Medicine
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
#
from __future__ import print_function
import os
from EMAN2 import *
from EMAN2_utils import *
import EMAN2_utils
from sys import argv
from optparse import OptionParser
import sys
import numpy as np
#from scipy.stats import norm

def main():

	progname = os.path.basename(sys.argv[0])
	usage = """Allocates movie stack images from tiltseries to one directory per tiltseries, fixes the apix on the header, and automatically performs motion correction with MotionCor2 and tiltseries compliation"""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)
	
	parser.add_argument("--anglesindxinfilename",type=int,default=None,help="""Default=None. The filename of the images will be split at any occurence of the following delimiters: '_', '-', '+', '[' , ']' , ',' , ' ' (the two last ones are a comma and a blank space). Provide the index (position) of the angle in the split filename. For example, if the filename of an image is "my_specimen-oct-10-2015_-50_deg-from_k2 camera.mrc", it will be split into ['my','specimen','oct','10','2015','','50','deg','from','k2','camera','mrc']. The angle '-50', is at position 6 (starting from 0). Therefore, you would provide --anglesindxinfilename=6, assuming all images to be stacked/processed are similarly named. No worries about the minus sign disappearing. The program will look at whether there's a minus sign immediately preceeding the position where the angle info is.""")
	parser.add_argument("--apix", type=float,default=0.0,help="""Default=0.0. Set/reset the apix value on the header of files.""")
	
	parser.add_argument("--donotclean",action="store_true",default=False,help="""Default=False. Prevent deletion of some files. By default, potentially garbage files containing the following strings between asterisks will be removed: *Back* *Center* *Track* *xml.* *shifts*'""")

	parser.add_argument("--gpu", type=str,default='0',help="Default=0 (first GPU used by default). Which GPU to use if --motioncorr is provided. To request multiple GPUs, list them without commas. For example, --gpu=0 2 4 5, would use the indicated GPUs.""")
	
	parser.add_argument("--idposition", type=int,default=1,help="Default=0 (no to used). The position in the filenames at which the identifying id (usually a numbered series across files) occurs. Position is defined by separator characters, such as underscores and hyphens. For example, for a series of files 'series_01_anglex.mrc', 'series_01_angley.mrc', 'series_02_anglex.mrc', 'series_02_angley.mrc', etc., providing id=1 would separate the files based on the labels '01' and '02', since they occur at 'position 1', starting from 0, where 'series' is the string occuring at position 0.")

	parser.add_argument("--mdoc",action="store_true",default=False,help="""Default=False. Provide this parameter to use mdocs present in the same directory as the image files to process.""")
	parser.add_argument("--motioncorr",action="store_true",default=False,help="""Default=.False. Run motion correction after changing the apix value on the header if --apix is provided""")
	
	parser.add_argument("--ppid", type=int, help="Default=1. Set the PID of the parent process, used for cross platform PPID",default=-1)
	
	parser.add_argument("--stem", type=str, default=None, help="""Default=None. Provide the common string at the beginning of all images in the directory, before the identifying label. To select a subset of files, provide the common string present in the subset of files to process. For example, in a series of files 'tomo_01_xyshsj.mrc, 'tomo_01_sfasdfd.mrc', 'tomo_02_asfds.mrc', if you provide --stem=tomo_, all files will be processed, but if you provide --stem=tomo_01, only the files with this labeled will be processed while the one with the 'tomo_02' label will be ignored.""")
	
	parser.add_argument("--tag",type=str,default=None,help=""""Default=None. String to append to the beginning of the tiltseries output filename. The default is filename is '<stem>_stack.st' where <stem> is whatever you feed to --stem; if tag=xxx, the output will be '<stem>_xxx_stack.st' """)	
	
	parser.add_argument("--tiltstacker_path",type=str,default=None,help=""""Default=None. Path to e2tomo_tilstacker.py version to use if you don't want to use the one from the EMAN2 installation.""")	

	parser.add_argument("--verbose", "-v", type=int, default=0, help="Default 0. Verbose level [0-9], higner number means higher level of verboseness",dest="verbose", action="store", metavar="n")

	(options, args) = parser.parse_args()
	
	logger = E2init(sys.argv, options.ppid)

	c=os.getcwd()
	findir=os.listdir(c)

	imgs=[]
	mdocs = []
	tids=set([])
	
	directories=[]

	index = options.idposition

	if not options.donotclean:
		try:
			os.system('rm *Back* *Center* *Track* *xml.* *shifts*')
		except:
			pass

	extensions =  EMAN2_utils.extensions()

	for f in findir:
		fstem,ext = os.path.splitext(os.path.basename(f))

		#if '.mrc' in f[-4:] and options.stem in f:
		if options.stem in f:
			if ext in extensions:
				imgs.append(f)
				#tid=f.split('tomo_')[-1].split('_')[0]
				tid= f.split('_')[index].split('.')[0]		
				tids.add(tid)
			if '.mdoc' in f[-5:]:
				mdocs.append(f)		
	
	mdocs.sort()

	tids=list(tids)
	tids.sort()

	tidsdict={}
	for tid in tids:
		fulltid=tid
		numstid=tid
		#print('\nbefore cleanup numstid={}'.format(numstid))

		for ch in tid:
			if ch.isalpha():
				numstid=numstid.replace(ch,'')
		#print('\nafter cleanup numstid={}'.format(numstid))
		if not numstid:
			tids.remove(tid)
		elif numstid:
			try:
				num=int(numstid)
				tidsdict.update({num:fulltid}) #for example for 'ts0005', this will be {5:ts0005}
			except:
				tids.remove(tid)
				print('\nWARNING: aberrant ID={} REMOVED'.format(tid))



	print("\n(e2allocate_files)(main)!!!!!!!!!!!!!!!! tids are tids={}".format(tids))
	directories=[]
	directories_dict={}
	if tids:
		directories,directories_dict=distributefiles(options,tidsdict,imgs,mdocs)
		
	elif not tids:
		if options.verbose:
			print("\n(e2allocate_files)(main) ERROR: no files to distribute into directories. Looking for directories now.")
			sys.exit(1)
		#directories = [f for f in findir if os.path.isdir(f)]
		
	jj=0
	if directories and directories_dict:
		directories.sort()
	
		try:
			os.mkdir('tiltseries_compiled')
		except:
			print('\ndirectory "tiltseries_compiled" exists already')
		
		cmdsmv=['mv ' + img + ' mrcs/' for img in imgs]
		for cmd in cmdsmv:
			runcmd(options,cmd)


		for d in directories:
			dpath = c + "/" + d
			findird = os.listdir(dpath)
			k=0
			stacks=''
			fixapix=False
			num = d.replace('t','')
			stem = options.stem
			intag = ""
			if options.tag:
				intag=options.tag
				print("\ninitial intag={}".format(intag))

				if "_" not in intag[-1:]:
					intag+="_"
				print("\nfinal intag={}".format(intag))
			if "_" not in stem[-1:]:
				stem+="_"
				#tag = " --tag " + options.stem + "_" + str(num)
			
			#tag = stem + intag + str(num)

			tag = stem + directories_dict[d]
			#if "_" not in tag[-1:]:
			#	tag+="_"
			print("\ntherefore final tag={}".format(tag))
			
			#outfile = ".st"
			#if options.tag:
			outfile = tag.rstrip('_') + ".st"
			
			print("\noutfile={}".format(outfile))
			#sys.exit(1)

			for f in findird:
				try:
					hdr=EMData(dpath+'/'+f,0,True)		
					nframes=hdr['nz']
				
					apixhdr=float(hdr['apix_x'])
					if options.apix and round(float(apixhdr),2) != round(float(options.apix),2):
						fixapix=True
						break #it only takes ONE image having the incorrect apix to turn fixapix on
				except:
					pass

			cmd='cd ' + dpath		
			if fixapix:
				cmdapix=' && e2procheader.py *.mrc --stem apix --valtype float --stemval ' + str(options.apix) 		
				cmd+=cmdapix
				print("\nfixing apix for frames in dir={}\ncmdapix is {}".format(d,cmdapix))
			
			if options.motioncorr:	
				#try:
				#	os.mkdir(dcorr)
				#except:
				#	print('\nerror trying to make new directory for motion corrected frames, dir={}'.format(dcorr))
			 
				#makenewdir(options,dcorr,'_aligned')
			
				#cmdmotioncorr = " && cd " + c + " && MotionCor2 -InMrc ./" + d + "/ -OutMrc " + dcorr + "/ -Patch 7,7,20 -Align 1 -LogFile -Iter 6 -Tol 0.5 -Throw 0 -Gpu " + str(options.gpu) + " -Serial 1 -PixSize " + str(options.apix) + " -Mag 1 -InFmMotion 1"

				cmdmotioncorr=" && for i in ./*.mrc; do MotionCor2 -InMrc $i -OutMrc ${i%.mrc}_aligned.mrc -LogFile ${i%_aligned.log} -Patch 7 7 20 -Iter 6 -Tol 0.50 -Bft 500.0 150.00 -StackZ " + str(nframes) + " -Align 1 -Mag 1 -InFmMotion 1 -GpuMemUsage 0.50 -Gpu " + options.gpu

				if options.apix:
					cmdmotioncorr += " -PixSize " + str(options.apix)
				
				cmd+=cmdmotioncorr + ";done"
				cmd += " && mkdir raw aligned logs && mv *.log logs/ && mv *aligned.mrc aligned/ && mv *.mrc raw/ && cd aligned/"
				print("\ncorrecting motion for frames in dir={}\ncmdmotioncorr is {}".format(d,cmdmotioncorr))
			
				#cmd += " && e2tomo_tiltstacker.py --input .mrc --anglesindxinfilename " + str(options.anglesindxinfilename) + tag
				#num = d.replace('t','')
				#cmd += " && cp tomostacker_01/" + outfile + " ../../tiltseries_compiled/" + outfile
				#cmd += " && cp tomostacker_01/" + outfile.replace(".st",".rawtlt") + " ../../tiltseries_compiled/" + outfile.replace(".st",".rawtlt")
			
			#elif not options.motioncorr:
				#cmd += " && e2tomo_tiltstacker.py --input .mrc --anglesindxinfilename " + str(options.anglesindxinfilename) + tag
				#num = d.replace('t','')
				#cmd += " && cp tomostacker_01/stack.st ../tiltseries_compiled/stack_" + str(num) + ".st"
				#cmd += " && cp tomostacker_01/stack.rawtlt ../tiltseries_compiled/stack_" + str(num) + ".rawtlt"
			

			if options.tiltstacker_path:
				cmd += " && python3 " + options.tiltstacker_path + " --input .mrc --tag " + tag
			else:
				cmd += " && e2tomo_tiltstacker.py --input .mrc --tag " + tag
			if options.anglesindxinfilename:
				cmd += " --anglesindxinfilename " + str(options.anglesindxinfilename) 
			if options.mdoc:
				cmd += " --mdoc " + options.stem #e2tomo_tiltstacker.py will look for .doc files that contain --stem automatically.
			
			cmd += " && cp " + dpath + "/tomostacker_01/" + os.path.basename(outfile) + " " + c + "/tiltseries_compiled/" + outfile
			rawtltf = os.path.basename(outfile).replace(".st",".rawtlt")
			cmd += " && cp " + dpath + "/tomostacker_01/" + rawtltf + " " + c + "/tiltseries_compiled/" + rawtltf
			cmd += " && mkdir " + dpath + "/tomostacker_01/recon "
			cmd += " && ln -s " + dpath + "/tomostacker_01/" + os.path.basename(outfile) + " " + dpath + "/tomostacker_01/recon/" + os.path.basename(outfile)
			cmd += " && cp " + dpath + "/tomostacker_01/" + rawtltf + " " + dpath + "/tomostacker_01/recon/" + rawtltf
			
			if cmd:
				
				print("\nrunning cmd={}".format(cmd))
				#pass
				runcmd(options,cmd)
			else:
				if options.verbose: print("\nWARNING: cmd={} is empty".format(cmd))
			
			jj+=1
			#if jj>0:
			#	break
		
		#if not options.donotclean:




					
	elif not directories:
		print("\nWARNING: could not find any directories with images to process. EXITING.")
		sys.exit(1)
		
	E2end(logger)
	return
	
	
def distributefiles(options,tidsdict,imgs,mdocs):
	#tids=list(tids)
	#print('\n(e2allocate_files.py)(distributefiles) tids are',tids)
	#tids.sort()
	#print("\n(e2allocate_files.py)(distributefiles) sorted", tids)
	#tids.reverse()
	#print("\n(e2allocate_files.py)(distributefiles) reversed", tids)
	#imgs.reverse()
	

	try:
		os.mkdir('mrcs')
	except:
		print('\ndirectory "mrcs" exists already')

	nums=list(tidsdict.keys())
	print("\nnumstype={} and nums={}".format(type(nums),nums))
	nums.sort()
	zfillfactor=0
	directories = set([])
	directories_dict = {}
	#try:
	#	nums=[int(tid) for tid in tids]
	#except:
	#	nums=[int(tid.replace(j,'')) for tid in tids for j in tid if j.isalpha()]
	
	if nums:
		print("\nnums are {}".format(nums))
		#sys.exit(1)
		zfillfactor = len(str(max(nums)))
	else:
		print('\n(e2allocate_files)(distributefiles) ERROR: empty values for zfillfactor={} and nums={}'.format(zfillfactor,nums))
		sys.exit(1)

	print("\n!!!!!!!!zfillfactor is {}".format(zfillfactor))
	
	#for tid in tids:
	mdocs_found = {}
	for num in nums:
		#findir = os.listdir()
		#tid = str(tid)
		directory='t'+str(num).zfill(zfillfactor)
		directories.add(directory)
		
		makenewdir(options,directory)
		print("\nanalyzing potential files for candidate directory={}".format(directory))
		tidorig = tidsdict[num] 
		tid = '_' + tidsdict[num] + '_'
		#id=options.stem+str(num)+'_'
		
		directories_dict.update({directory:tidorig})
		#toremove=set([])
		for img in imgs:
			stem,ext = os.path.splitext(os.path.basename(img))
			if options.verbose > 5:
				print("\n(e2allocate_files.py)(distributefiles) looking for id={} in img={}".format(tid,img))
	
			if tid.replace('_','') in img.split('_')[options.idposition]:
				if options.verbose:
					print("\n(e2allocate_files.py)(distributefiles) found img={} with id={}".format(img,tid))
					print("\n(e2allocate_files.py)(distributefiles) I will create a symlink from img={} to newlocation={}".format(img,directory+'/'+img))

				if img not in os.listdir(os.getcwd()+'/mrcs'):
					try:
						os.rename(os.getcwd() + "/" + img, os.getcwd() + '/mrcs/' + img)
						try:
							os.symlink(os.getcwd() + '/mrcs/' + img, os.getcwd() + "/" +directory + '/' + img)
						except:
							print(f'\nfailed to create symlink for img={img} in mrcs directory into specific directory={directory}')
							sys.exit(1)
					except:
						if img in os.listdir(os.getcwd()+'/mrcs'):
							print(f'\nimg={img} is already in mrcs directory')
						else:
							print(f'\nfailed to copy img={img} into mrcs directory')
							sys.exit(1)
				else:
					print(f'\nimg={img} is presumablt already in mrcs directory')

				#os.symlink(os.getcwd() + "/" + img, os.getcwd() + "/" +directory + '/' + img)
				#toremove.add(img)
				try:
					tltfile=img.replace(ext,'.rawtlt')
					os.rename(tltfile,directory+'/'+tltfile)
				except:
					if options.verbose > 5:
						print("\n(e2allocate_files.py)(distributefiles) no .rawtlt file found for img {}".format(img))
				try:				
					txtfile=img.replace(ext,'.txt')
					os.rename(txtfile,directory+'/'+txtfile)
				except:
					if options.verbose > 5:				
						print("\n(e2allocate_files.py)(distributefiles) no .txt file found for img {}".format(img))
	
		if options.mdoc:
			if mdocs:
				print("\n(e2allocate_files)(distributefiles) found these mdocs={}".format(mdocs))
				for mdoc in mdocs:

					if tidorig in mdoc: #c: mdoc might end with tid.mdoc instead of tid_.mdoc, yet tid includes leading and trailing _ for other purposes below; thus here we use tidorig instead 
						print("\n(e2allocate_files)(distributefiles) for tidorig={} found mdoc={}".format(tidorig,mdoc))
						#cmd_mv_mdoc = "mv " + os.getcwd() + "/" + mdoc+ " " + os.getcwd() + "/" +directory + '/' + mdoc
						#runcmd(options,cmd_mv_mdoc)

						mdoc_source = os.getcwd() + "/" + mdoc
						mdoc_destination = os.getcwd() + "/" +directory + '/' + mdoc
						os.rename(mdoc_source,mdoc_destination)
						print("copied mdoc from source={} to destination={}".format(mdoc_source,mdoc_destination))
						mdocs_found.update({tidorig:mdoc})
					else:
						print("\nWARNING: found mdoc={} but tidorig={} is not in the filename".format(mdoc,tidorig))
						#sys.exit(1)
						pass
			else:
				print("\n\n\n(e2allocate_files)(distributefiles) WARNING: no mdocs files found in this directory even though --mdoc flag was provided.")

	for idm in mdocs_found:
		if mdocs_found[idm]:
			pass
		elif not mdocs_found[idm]:
			print("\nfor idm={} could NOT find an mdoc".format(idm))


	directories = list(directories)
	return directories,directories_dict


def makenewdir(options,d,key='!@#$%^&*()_+'):
	c=os.getcwd()
	findir=os.listdir(c)
	try:
		if d not in findir:
			#print("\n(e2allocate_files.py)(makenewdir) key={}; if this key is NOT in the directory filename, the directory SHOULD be made.".format(key))
			if d.count(key) > 1:
				print("\n(e2allocate_files.py)(makenewdir) skipping creation of directory={}, since it seems aberrant".format(d))	
			elif d.count(key) <= 1:
				os.mkdir(d)
				print("\n(e2allocate_files.py)(makenewdir) created directory={}".format(d))
				
		else:
			if options.verbose > 5:
				print("\n(e2allocate_files.py)(makenewdir) directory={} exists".format(d))
	except:
		print("\n(e2allocate_files.py)(makenewdir) directory={} likely exists".format(d))

	return
	
		
if __name__ == '__main__':
	main()
