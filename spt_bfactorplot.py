#!/usr/bin/env python
#
# Author: Jesus Galaz, 22/Jan/2017; last update Mar/2023

import os
from EMAN2 import *
import sys
import numpy
import math
import collections
import time
from datetime import timedelta
import concurrent.futures

	
def main():
	start = time.perf_counter()

	progname = os.path.basename(sys.argv[0])
	usage = """This program computes a pseudo-b-factor plot using an aligned even and odd subtomogram average stacks, to determine convergance and plateuing of resolution."""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)
	
	parser.add_argument("--apix",type=float,default=0.0,help="""Default=0.0 (not used). Use this apix value where relevant instead of whatever is in the header of the reference and the particles.""")
	parser.add_argument("--averager",type=str,default="mean.tomo",help="""Default=mean.tomo. The type of averager used to produce the class average.""")

	parser.add_argument("--inputeven", type=str, default='',help="""Default=None. The name of the EVEN aligned volume stack after gold-standard SPT refinement. MUST be HDF since volume stack support is required.""")
	parser.add_argument("--inputodd", type=str, default='',help="""Default=None. The name of the ODD aligned volume stack after gold-standard SPT refinement. MUST be HDF since volume stack support is required.""")
	
	parser.add_argument("--mask_file",type=str,default=None,help="""Default=None. Mask file to multiple particles by.""")
	parser.add_argument("--mask_radius",type=int,default=None,help="""Default=None. Provide the radius of a softmask to apply to the particles""")

	parser.add_argument("--path",type=str,default='sptbfactor',help="""Default=spt. Directory to store results in. The default is a numbered series of directories containing the prefix 'spt'; for example, spt_02 will be the directory by default if 'spt_01' already exists.""")
	parser.add_argument("--ppid", type=int, help="Set the PID of the parent process, used for cross platform PPID",default=-1)

	parser.add_argument("--savesteps",action='store_true',default=False,help="""Save intermediate averages.""")

	parser.add_argument("--step",type=int,default=1,help="""Default=1. Number of particles to increase from one data point to the next. For example, --step=10 will compute the B-factor averaging 10 particles from the even set and 10 from the odd set; then 20; then 40; etc.""")
	parser.add_argument("--subset",type=int,default=None,help="""Default=None. Subset of particles to run the analysis on""")
	parser.add_argument("--sym", type=str, default='', help="""Default=None (equivalent to c1). Symmetry to impose -choices are: c<n>, d<n>, h<n>, tet, oct, icos""")
	
	parser.add_argument("--threads",type=int,default=12,help="""Default=12.""")

	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n",type=int, default=0, help="verbose level [0-9], higner number means higher level of verboseness")
	
	(options, args) = parser.parse_args()	#c:this parses the options or "arguments" listed 
											#c:above so that they're accesible in the form of option.argument; 
											#c:for example, the input for --template would be accesible as options.template
		
	logger = E2init(sys.argv, options.ppid)	#c:this initiates the EMAN2 logger such that the execution
											#of the program with the specified parameters will be logged
											#(written) in the invisible file .eman2log.txt
	
	options.averager=parsemodopt(options.averager)
	
	from EMAN2_utils import makepath
	options = makepath(options,'sptbfactor')

	ne = EMUtil.get_image_count( options.inputeven )
	no = EMUtil.get_image_count( options.inputodd )

	nfinal = (ne+no)//2
	if ne < nfinal:
		nfinal = ne
	elif no < nfinal:
		nfinal = no

	if options.subset:
		if options.subset >= nfinal:
			print(f"\nWARNING: Ignoring --subset={options.subset}, since it must be < the lesser of the number of particles in even or odd stacks, n-even={ne}, n-odd={no}")
		else:
			if ne < 3 or no <3:
				print("\nERROR: to use --subset each of the even and odd particle stacks needs to have 3 or more particles in it")
				sys.exit(1)	
			else:
				nfinal = options.subset
				if nfinal%2:
					nfinal -= 1
				no = nfinal
				nw = nfinal

	if options.savesteps:
		for i in range( nfinal//options.step ):
			a=EMData(8,8,8)
			a.write_image(options.path +'/tmpavgs_odd.hdf',i)
			a.write_image(options.path +'/tmpavgs_even.hdf',i)
		
	mask=None
	if options.mask_radius:
		mask=makemask(options)

	fscareas = []
	fscareasdict={}
	
	preve = EMData( options.inputeven, 0 )
	prevo = EMData( options.inputodd, 0 )
	
	avge = preve.copy()
	avgo = prevo.copy()
	
	data_dict = {}
	count = 0
	
	for nval in range( 1, nfinal+1, options.step ):
		data_dict.update({count:nval})
		count+=1

	results=parallelization_threads_avg(options,data_dict,mask)

	results_ordered = collections.OrderedDict(sorted(results.items()))

	x=0
	fsclines = []
	for k, fscarea in results_ordered.items():
		f = open( options.path +'/n_vs_fsc.txt','w' )
		fscline = str(x) + '\t'+ str(fscarea) + '\n'
		fsclines.append( fscline )
		x+=1
	
	f.writelines( fsclines )
	f.close()

	elapsed = time.perf_counter() - start
	print(str(timedelta(seconds=elapsed)))

	return


def data_point_generator(options,nval,mask):
	avge = averagerfunc( options.inputeven, options, nval )
	avgo = averagerfunc( options.inputodd, options, nval )
	
	if mask:
		avge.process_inplace('mask.soft',{'outer_radius':radius_expanded})
		avgo.process_inplace('mask.soft',{'outer_radius':radius_expanded})

	if options.sym:
		avge.process_inplace('xform.applysym',{'sym':options.sym})
		avgo.process_inplace('xform.applysym',{'sym':options.sym})

	index = nval//options.step
	if options.savesteps:	
		avgo.write_image(options.path +'/tmpavgs_odd.hdf',index)
		avge.write_image(options.path +'/tmpavgs_even.hdf',index)

	fscarea = calcfsc( options, avge, avgo )

	return nval,fscarea,index


def makemask(options):
	hdr = EMData( options.inputeven, 0, True )
	box = hdr['nx']
	
	if options.vserbose:
		print(f"\nradius = {options.mask_radius}")

	mask = EMData(box,box,box)
	mask.to_one()
	if options.mask_radius:
		mask.process_inplace('mask.soft',{'outer_radius':options.mask_radius})
	elif options.mask_file:
		maskf = EMData(options.mask_file)
		if maskf['nx'] != mask['n'] or maskf['ny'] != mask['ny'] or maskf['nz'] != mask['nz']:
			maskf=clip3d(maskf,box)
			print(f"\nWARNING: --mask_file={options.mask_file} nx{nx}, ny={ny}, nz={nz} not equal to input data's nx={box}, ny={box}, nz={box}")
			print("\nclipping --mask_file to match data's box size.")
		mask *= maskf 

	return mask


def averagerfunc(stack,options,nval):
	if options.verbose:
		print(f"\ncomputing average with n={nval} particles, from file {stack}")
	
	avgr = Averagers.get( options.averager[0], options.averager[1])
	indexes = list(range(nval))
	indexes.sort()
	for i in indexes:
		ptcl = EMData( stack, i )
		nz=ptcl['nz']
		if options.verbose > 8:
			print(f"\nptcl['nz']={nz}")
	
		avgr.add_image( ptcl )
		if options.verbose > 8:
			print(f"\nadded ptcl {i}/{nval}")
	
	avg = avgr.finish()
	if avg:
		#avg.process_inplace('normalize.edgemean')
		
		return avg
	else:
		print(f"\nERROR: average with these many particles n={nval} failed")
		sys.exit(1)
		return


def calcfsc( options, img1, img2 ):
	
	img1fsc = img1.copy()
	img2fsc = img2.copy()
	
	apix = img1['apix_x']
	if options.apix:
		apix=options.apix
	
	fsc = img1fsc.calc_fourier_shell_correlation( img2fsc )
	third = len( fsc )//3
	xaxis = fsc[0:third]
	fsc = fsc[third:2*third]
	saxis = [x/apix for x in xaxis]

	fscfile = options.path + '/tmpfsc.txt'
	Util.save_data( saxis[1], saxis[1]-saxis[0], fsc[1:-1], fscfile )

	f=open(fscfile,'r')
	lines=f.readlines()
	fscarea = sum( [ float(line.split()[-1].replace('\n','')) for line in lines ])
	
	return fscarea


def parallelization_threads_avg(options,data_dict,mask):
	
	if options.verbose > 8:
		print(f"(spt_bfactorplot)(parallelization_threads_avg) type(data_dict_={type(data_dict)}")
		print(f"\n(parallelization_threads) threads={options.threads}")

	futures = []
	results = {}

	with concurrent.futures.ProcessPoolExecutor( max_workers=int(options.threads) ) as executor:
		counter=0
		for count in data_dict:
			nval=data_dict[count]
			future = executor.submit( data_point_generator, options, nval, mask)
			#futures.update({i:future})
			futures.append(future)
			if options.verbose > 8:
				print(f"\nadded future {counter}")
			counter+=1
		
		for future in concurrent.futures.as_completed(futures):
			nval,fscarea,index = future.result()
			results.update({nval:fscarea})
			if options.verbose > 8:
				print(f"\ngot result number {index}")

		if options.verbose > 8:
			print(f"\n(e2spt_simulation)(parallelization_threads_subtomosim) results={results}")

	return results

	
if '__main__' == __name__:
	main()
