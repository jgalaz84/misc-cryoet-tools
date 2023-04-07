#!/usr/bin/env python
#====================
#Author: Jesus Galaz-Montoya sep/2019 , Last update: mar/2023
#====================

from __future__ import print_function
from __future__ import division
from past.utils import old_div
from builtins import range
from EMAN2_utils import *
import EMAN2_utils
from EMAN2 import *
import sys
import math
import re
import numpy
import time
from datetime import timedelta


def main():
	start = time.perf_counter()
	
	usage = """tomo_tilt_stacker.py <imgs> <options> . 
	The options should be supplied in "--option=value" format, 
	replacing "option" for a valid option name, and "value" for an acceptable value for that option. 
	
	The program uses EMAN2 functions to stack individual .dm3, .tiff, .mrc, or .hdf images into an .mrc (or .st) stack,
	by supplying a common string to all the images to stack.
	
	For example:
	tomo_tilt_stacker.py --input=my_imgs --anglesindxinfilename=6

	It also generates a .rawtlt file with tilt angle values if --lowerend, --upperend and --tiltstep are provided.
	"""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)	

	parser.add_argument("--anglesindxinfilename",type=int,default=None,help="""Default=None. The filename of the images will be split at any occurence of the following delimiters: '_', '-', '+', '[' , ']' , ',' , ' ' (the two last ones are a comma and a blank space). Provide the index (position) of the angle in the split filename. For example, if the filename of an image is "my_specimen-oct-10-2015_-50_deg-from_k2 camera.mrc", it will be split into ['my','specimen','oct','10','2015','','50','deg','from','k2','camera','mrc']. The angle '-50', is at position 6 (starting from 0). Therefore, you would provide --anglesindxinfilename=6, assuming all images to be stacked/processed are similarly named. No worries about the minus sign disappearing. The program will look at whether there's a minus sign immediately preceeding the position where the angle info is.""")
	parser.add_argument("--apix",type=float,default=0.0,help="""True apix of images to be written on final stack.""")
	
	parser.add_argument("--calc_image_stats", action='store_true',default=False,help="""Default=False. Compute the distribution of min, max, mean, and sigma across images to see if there are anomalous ones.""")

	parser.add_argument("--exclude_extremes",action='store_true',default=False,help="""Default=False. Requires --calc_image_stats. Will exclude images with a mean value 3 standard deviations away from the "mean of means" of all images.""")

	parser.add_argument("--highesttilt",type=float,default=0.0,help="""Highest tilt angle. If not supplied, it will be assumed to be 1* --tiltrange.""")
	
	parser.add_argument("--input", type=str, default=None, help="""String common all files to process. For example, to process all .mrc files in a directory, you would run tomo_tilt_stacker.py --input=.mrc <parameters>.""")

	parser.add_argument("--lowesttilt",type=float,default=0.0,help="""Lowest tilt angle. If not supplied, it will be assumed to be -1* --tiltrange.""")
	
	parser.add_argument("--mdoc", type=str, default=None, help="""Unsorted .mdoc file to derive the angles and order of the tiltseries from if angles are not in the image filenames. This will also generate a defocuses file.""")

	parser.add_argument("--path",type=str,default='tomostacker',help="""Directory to store results in. The default is a numbered series of directories containing the prefix 'sptstacker'; for example, sptstacker_02 will be the directory by default if 'sptstacker_01' already exists.""")
	parser.add_argument("--ppid", type=int, help="Set the PID of the parent process, used for cross platform PPID",default=-1)
	parser.add_argument("--precheckfiles",action="store_true",default=False,help=""""Make sure that only valid images found by --input=* are processed -if unreadable or bad images are fed to the program, it might crash.""")

	parser.add_argument("--stackregardless",action="store_true",default=False,help=""""Stack images found with the common string provided through --stem2stack, even if the number of images does not match the predicted number of tilt angles.""")

	parser.add_argument("--tag",type=str,default=None,help=""""String to append to the beginning of the tiltseries output filename. The default is filename is 'stack.st'; if tag=xxx, the output will be 'xxx_stack.st' """)	
	parser.add_argument("--tiltrange",type=float,default=0.0,help="""If provided, this will make --lowesttilt=-1*tiltrange and --highesttilt=tiltrange. If the range is asymmetric, supply --lowesttilt and --highesttilt directly.""")
	parser.add_argument("--tiltstep",type=float,default=0.0,help="""Step between tilts. Required if using --stem2stack.""")
	
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n",type=int, default=0, help="verbose level [0-9], higher number means higher level of verboseness.")

	(options, args) = parser.parse_args()	
	
	#c: Check for sensible selected options
	if not options.anglesindxinfilename and not options.mdoc:
		if options.tiltrange and options.tiltstep:
			pass
		elif not options.tiltstep and not options.tiltrange:
			print("\nERROR: in the abscence of --tiltstep and --tiltrange, you must provide either --anglesindxinfilename or --mdoc")
			sys.exit(1)

	if options.verbose>8:
		print("\n--input={}".format(options.input))

	stem =  os.path.basename( options.input.replace('*','') )
	if options.verbose>8:
		print("\ntherefore, stem={}".format(stem))

	dirname = os.path.dirname( options.input.replace('*','') )
	if not dirname:
		dirname='.'

	options.path = dirname + "/" + options.path

	files2process_raw = fsindir(dirname)
	if options.verbose>8:
		print("\nfound files2process_raw={}".format(files2process_raw))

	#c: Exclude common files often in the same directory as images that need stacking, and include files only if they have a valid extension
	extensions = EMAN2_utils.extensions()

	files2process = [dirname + '/' + f for f in files2process_raw if stem in f and '.txt' not in f[-4:] and '.' not in f[0] and '.mdoc' not in f[-5:] and f.lower().endswith( tuple(extensions) )]

	#c: test for sanity of candidate images by attempting to read the header
	files2process_clean = []
	#for f in files2process:
	#	try:
	#		hdr = EMData(f,0,True)
	#		files2process_clean.append(f)
	#	except:
	#		print("\nSkipping file={} because it doesn't seem to be a valid image".format(f)) 		
	#
	#if options.precheckfiles:
	#	good=[]
	bad=[]
	for g in files2process:
		try:
			hdr=EMData(g,0,True)
			files2process_clean.append(g)
		except:
			bad.apppend(g)
	if files2process_clean:
		if options.verbose:
			print("Found n={} candidate files to process".format(files2process_clean))
	else:
		print("\nERROR: could not find any readable image files with the common string --input={}".format(options.input))
		sys.exit(1)
	if bad:
		print("\nWARNING: excluding the following n={} files because they are not readable (this happens sometimes with 'empty' image files)".format(len(bad)))
		for b in bad:
			print(b)


	files2process_clean.sort(key = lambda s: float(re.search('(\+|-)?\d+(\.\d+)?', s).group()))

	if options.verbose:
		print("\n(tomo_tilt_stacker.py)(main) files2process_clean and sorted={}".format(files2process_clean))

	if options.lowesttilt == 0.0 and options.tiltrange:
		options.lowesttilt = -1 * round(float(options.tiltrange),2)
		
	if options.highesttilt == 0.0 and options.tiltrange:
		options.highesttilt = round(float(options.tiltrange),2)
	
	#c: log the run for record keeping
	print("\nLogging")
	logger = E2init(sys.argv, options.ppid)

	print("\n(tomo_tilt_stacker.py)(main) These many img files were found n={} in directory={}".format( len(files2process_clean), dirname) )		
	
	#c: make a directory to store the stacked tilt series
	options = makepath( options, 'tomostacker' )

	#c: compute statistics if requested, as a proxy for potential outliers ("bad images") to remove
	means=[]
	sigmas=[]
	minima=[]
	maxima=[]

	if options.calc_image_stats:
		stats_dict = calc_image_stas(files2process_clean)

		if options.exclude_extremes:
			mean_avg = stats_dict['global_mean'][0]
			mean_std = stats_dict['global_mean'][1]

			#files2process_good = []
			for f in files2process_clean:
				img = EMData( f, 0 )
				mean = img['mean']
				sigma = img['sigma']
				minimum = img['minimum']
				maximum = img['maximum']

				if mean > mean_avg - 3.0*mean_std and mean < mean_avg + 3.0*mean_std:
					if options.verbose>8:
						print("\nkeeping image {}".format(f))
				else:
					files2process_clean.remove(f)
					if options.verbose:
						print("\nWARNING: potentially bad image={} excluded; mean={}, mean_avg={}, sigma={}, min={}, max={}".format(f, mean, mean_avg, sigma, minimum, maximum))

	kk=0
	
	if options.verbose>5:
		print("\n(tomo_tilt_stacker.py)(main) organizing tilt imgs")
	intiltsdict = organizetilts( options, files2process_clean )				#Create a dictionary in the form { indexintiltseries:[ tiltfile, tiltangle, damage_rank ]},
	if options.verbose>5:
		print("\n(tomo_tilt_stacker.py)(main) done organizing tilt imgs")	#where damage_rank tells you the order in which the images 
																			#were acquired regardless of wether the tilt series goes from 
																			#-tiltrange to +tiltrange, or 0 to -tiltrange then +tiltstep to +tiltrange, 
																			#or the opposite of these 
	outstackhdf = options.path + '/stack.hdf'
	if options.tag:
		outstackhdf = options.path + '/' + options.tag.rstrip('_') + '.hdf'

	if not intiltsdict:
		print("\n(tomo_tilt_stacker.py)(main) ERROR: intiltsdict is empty: {}".format(intiltsdict))
		sys.exit(1)
	else:
		if options.verbose > 5:
			print("\n(tomo_tilt_stacker.py)(main) intiltsdict is {}".format(intiltsdict))

	minindx = min(intiltsdict)
	if options.verbose>5:
		print("\n(tomo_tilt_stacker.py)(main) minindx is", minindx)
		print("\n(tomo_tilt_stacker.py)(main) getting image size from the image at the least tilted angle, intiltsdict[ minindx ][0]", intiltsdict[ minindx ][0])
	
	hdr = EMData( intiltsdict[minindx][0], 0, True )
	nx = hdr['nx']
	ny = hdr['ny']
	print(nx,ny)

	if options.verbose > 5:
		print("\n(tomo_tilt_stacker.py)(main) outstack is", outstackhdf)
		print("\n(tomo_tilt_stacker.py)(main) intiltsdict.keys() are={}, because intiltsdict is={}".format(intiltsdict.keys(),intiltsdict))
	
	finalindexesordered = [x for x in intiltsdict.keys()]
	finalindexesordered.sort()

	damagelist=[]


	for index in finalindexesordered:	
		intiltimgfile =	intiltsdict[index][0]
			
		if options.verbose > 9:
			print("\n(tomo_tilt_stacker.py)(main) at index {} we have image {}, collected in turn {}".format( index, intiltsdict[index][0], intiltsdict[index][-1] ))
		
		intiltimg = EMData( intiltimgfile, 0 )
		tiltangle = intiltsdict[index][1]
		intiltimg['tilt_angle'] = tiltangle
		
		damage_rank = intiltsdict[index][2]
		intiltimg['damage_rank'] = damage_rank

		damagelist.append(damage_rank)

		intiltimg.write_image( outstackhdf, -1 )
	
	orderfilepath = options.path + '/collection_order.txt'
	textwriter(damagelist,options,orderfilepath,invert=0,xvals=None,onlydata=True)

	tmp = options.path + '/tmp.hdf'

	if options.verbose>8:
		print("\n(tomo_tilt_stacker.py)(main) reading outstackhdf hdr, for file {}".format(outstackhdf))
	outtilthdr = EMData(outstackhdf,0,True)
	currentapix = outtilthdr['apix_x']
		
	outstackst = outstackhdf.replace('.hdf','.st')

	if options.verbose>8:
		print("\n\n\n\nLAST instance of outstackhdf={}".format(outstackhdf))
		print("\nbecuase tag={}".format(options.tag))
	
	stcmd = 'e2proc2d.py ' + outstackhdf + ' ' + outstackst + ' --twod2threed'
	
	if options.verbose>8:
		print("\n(tomo_tilt_stacker.py)(main) stcmd is", stcmd)	

	runcmd(options,stcmd)
	os.remove(outstackhdf)

	findir = os.listdir(options.path)
	for f in findir:
		if '~' in f:
			if options.verbose:
				print("\n(tomo_tilt_stacker.py)(main) file to remove",f)
				print("in path", options.path + '/' + f)
			os.remove(options.path + '/' + f)

	E2end( logger )

	elapsed = time.perf_counter() - start
	print(str(timedelta(seconds=elapsed)))
	
	return


def calc_image_stas(files2process_clean):
	for f in files2process_clean:
		#intiltimgfile =	intiltsdict[index][0]
		intiltimg = EMData( f, 0, True )
		try:
			means.append(intiltimg['mean'])
		except:
			try:
				means.append(intiltimg['MRC.mean'])
			except:
				print("\nWARNING: cannot get mean nor MRC.mean for img={}. May not be a valid image".format(f))			
		try:
			minima.append(intiltimg['minimum'])
		except:
			try:
				minima.append(intiltimg['MRC.minimum'])
			except:
				print("\nWARNING: cannot get minimum nor MRC.minimum for img={}. May not be a valid image".format(f))
		try:
			maxima.append(intiltimg['maximum'])
		except:
			try:
				maxima.append(intiltimg['MRC.maximum'])
			except:
				print("\nWARNING: cannot get minimum nor MRC.minimum for img={}. May not be a valid image".format(f))
			#stats_dict.update( { index:{'mean':mean,'sigma':sigma,'maximum':maximum,'minimum':minimum} } )
		try:
			sigmas.append(intiltimg['sigma'])
		except:
			print("\nWARNING: cannot get sigma for img={}. May not be a valid image".format(f))
	
	mean_avg = numpy.mean(means)
	sigma_avg = numpy.mean(sigmas)
	minimum_avg = numpy.mean(minima)
	maximum_avg = numpy.mean(maxima)
	
	mean_std = numpy.std(means)
	sigma_std = numpy.std(sigmas)
	minimum_std = numpy.std(minima)
	maximum_std = numpy.std(maxima)

	return {'global_mean':[mean_avg,mean_std],'global_sigma':[sigma_avg,sigma_std],'global_min':[minimum_avg,minimum_std],'global_max':[maximum_avg,maximum_std]}


def getangles( options, ntilts, raworder=False ):
			
	print("There was no .tlt file so I'll generate the angles using lowesttilt=%f, highesttilt=%f, tiltstep=%f" % (options.lowesttilt, options.highesttilt, options.tiltstep))
	generate = floatrange( options.lowesttilt, options.highesttilt, options.tiltstep )
	angles=[ round(float(x),2) for x in generate ]

	if ntilts:
		nangles=len(angles)
		dif=int(math.fabs( ntilts - nangles))
		if nangles < ntilts:
			supplementstart = angles[-1]+options.tiltstep
			supplementend = angles[-1] + dif*options.tiltstep
			angles += [ round(float(x),2) for x in floatrange( supplementstart, supplementend,  options.tiltstep ) ]
		elif nangles > ntilts:
			angles = angles[:-1*dif] #cut the angles list by however many exceeding elements there are, as indicated by dif
				
	
	print("BEFORE sorting, angles are", angles)
	
	print("raworder", raworder)
	print("not raworder", not raworder)

	
	if not raworder:
		angles.sort()
		angles.reverse()
		print("\n(tomo_tilt_stacker.py)(getangles) AFTER REVERSING (ordered from largest to smallest), angles are", angles)
	
	return angles


def writetlt( angles, options ):
	
	print("(writetlt) these many angles", len(angles))
	angless = list( angles )
	
	#if not raworder:
	#	angless.sort()
	#	angless.reverse()
		
	lines = []
	
	k=0			
	for a in angless:
		line = str(a) + '\n'
		lines.append( line )
		
		k+=1
	
	tltfilepath = options.path + '/stack.rawtlt'
	if options.tag:
		tltfilepath = options.path + '/' + options.tag.rstrip('_') + '.rawtlt'
		#print("""\n(writetlt) tltfilepath={}""".format(tltfilepath))
		#if "_" not in options.tag[-1:]:
		#	tltfilepath = options.path + '/' + options.tag + '_stack.rawtlt'

	textwriter(lines,options,tltfilepath,invert=0,xvals=None,onlydata=True)

	return


def organizetilts( options, intilts, raworder=False ):
	
	intilts_unsorted = list(set(intilts)) #make extra sure there are no duplicate files
	intilts = intilts_unsorted.copy()
	intilts.sort()

	intiltsdict = {}
	angles = []

	if options.anglesindxinfilename:
		collectionindex=0
		anglesdict = {}
		#print("\nthere are these many intilts n={}".format(len(intilts)))
		#print("\nand they are {}".format(intilts))
		for intilt in intilts:
			
			extension = os.path.splitext(os.path.basename(intilt))[-1]

			parsedname = intilt.replace(extension,'').replace(',',' ').replace('-',' ').replace('_',' ').replace('[',' ').replace(']',' ').replace('+',' ').split()
			#dividerstoadd = options.anglesindxinfilename - 1
			
			print('\n(tomo_tilt_stacker)(organizetilts) parsedname is',parsedname)
			angle = parsedname[options.anglesindxinfilename]
			try:
				angle = float( re.sub('[^\d\.\-]', '', parsedname[options.anglesindxinfilename]).rstrip('.') ) #remove strings from number except dots and minuses, and any leftover trailing dots
			except:
				print("\n(tomo_tilt_stacker)(organizetilts) invalid angle={}. make sure --anglesindxinfilename is correct! Remember, indexes start from 0".format(parsedname[options.anglesindxinfilename]))
				sys.exit(1)
			
			charsum = 0
			for i in range(options.anglesindxinfilename):
				charsum += len(parsedname[i])
				#if len(parsedname[i]) == 0:
				charsum += 1 
			#charsum += options.anglesindxinfilename

			#sign = intilt[charsum+1]
			sign = intilt[charsum]
			#print("\nby position %d sign is %s" % (charsum+1,sign))

			#print("angle is".format(angle))
			#print("sign is".format(sign))

			if sign == '-':
				angle *= -1
				if options.verbose:
					print("\n(tomo_tilt_stacker)(organizetilts) therfore corrected angle to be negative, now it is {}".format(angle))

			
			#sign2 = intilt.split(str(angle))[0][-1]
			#print "by other means, sign2 is",sign2
			
			anglesdict.update({angle:[intilt,collectionindex]})
			if options.verbose:
				print("\nadded collectionindex={} for angle={} and intilt={}".format(collectionindex,angle,intilt))
			if angle not in angles:
				angles.append( angle )
			collectionindex+=1			#since there could be parasitic images (two or more images per tilt angle due to failed attempts)
										#collectionindex needs to continue to increase even if the repeat images are excluded to properly dose weight later
		
		angles.sort()
		print("\nSORTED angles={}".format(angles))

		colindx_most_neg = anglesdict[min(angles)][-1]
		colindx_most_pos = anglesdict[max(angles)][-1]
		if colindx_most_pos < colindx_most_neg:
			print("\nWARNING: colindx_most_pos={} < colindx_most_neg={}. REVERSING angles".format(colindx_most_pos, colindx_most_neg))
			angles.reverse()
			print("\nREVERSED angles={}".format(angles))
		else:
			print("\ncolindx_most_pos={},colindx_most_neg={}".format(colindx_most_pos, colindx_most_neg))

		
		writetlt( angles, options )

		#kkkk=0
		finalangles = []
		
		#print('anglesdict is', anglesdict)
		indexintiltseries = 0
		newindexintiltseries = 0
		
		for angle in angles:
			#indexintiltseries = kkkk
			#newindexintiltseries = kkkk
			if options.verbose>5:
				print("\n(tomo_tilt_stacker)(organizetilts) examining angle {} corresponding to index {}".format(angle,indexintiltseries))	
			#{ indexintiltseries:[ imgile, angle, collectionindx]} 
			
			intiltsdict.update( { newindexintiltseries:[ anglesdict[angle][0], angle, anglesdict[angle][1] ]} )
			if options.verbose>5:
				print("\n(tomo_tilt_stacker)(organizetilts) updating intiltsdict with img={}, angle={}, order={}".format(anglesdict[angle][0], angle, anglesdict[angle][1] ) )
			finalangles.append(angle)
			
			if options.verbose>5:
				print("\n(tomo_tilt_stacker)(organizetilts) kept oldindex={}, which after excluding images became newindex={}".format(indexintiltseries,newindexintiltseries))
			newindexintiltseries+=1
			
		
			indexintiltseries+=1
		
		angles=list(finalangles)


	elif options.mdoc:

		stemf,extensionf = os.path.splitext(os.path.basename(options.mdoc))

		if not extensionf:
			mdocs = [ f for f in os.listdir(os.getcwd()) if '.mdoc' in f[-5:] and options.mdoc in f]
			truemdocs=[]
			if mdocs:
				if len(mdocs)>1:
					for mdoc in mdocs:
						if 'unsorted' in mdoc:
							truemdocs.append(mdoc)
					if len(truemdocs) > 1:
						print("\nERROR: there's more than one valid candidate file. Make sure to have only one valid mdoc file in this directory. mdocs={}".format(mdocs))
						sys.exit(1)
				elif len(mdocs) == 1:
					truemdocs.append(mdocs[0])
			else:
				print("\nERROR: --mdoc was provided a string that is not a complete file name, {}; attempts to find an mdoc file with this string failed; copy a valid mdoc file to this directory".format(options.mdoc))
				sys.exit(1)
			
			options.mdoc = truemdocs[0]
			
		stemf,extensionf = os.path.splitext(os.path.basename(options.mdoc))

		stemintilts,extensionintilts = os.path.splitext(os.path.basename(intilts[0]))

		with open(options.mdoc,'r') as f:
			bigls = [l for l in f.read().split('\n\n') if 'ZValue' in l]

			if options.verbose:
				print("\nnumber of entries for distinct images in mdocfile={} are len(bigls)={}".format(m,len(bigls)))

			stemf_root = stemf.split('.')[0]

			collectionlines = [ str(int( l.split("[ZValue = " )[-1].split(']\n')[0] ))+ "\t" + str(round(float( l.split("TiltAngle = " )[-1].split('\n')[0] ),2)) +"\n" for l in bigls ]

			#c: take a subset of the data matching len(intilts) because the mdoc may have been generated from another mdoc with a different number if images
			writef(collectionlines[:len(intilts)], options.path +'/' + stemf_root + ".collection")

			angles_and_defocuses_and_collection_unsorted = [ [round(float( l.split("TiltAngle = " )[-1].split('\n')[0] ),2), round(float( l.split("\nDefocus = " )[-1].split('\n')[0] ),2), int( l.split("[ZValue = " )[-1].split(']\n')[0] ), os.path.basename( l.split("SubFramePath = ")[-1].split("\n")[0] ) ] for l in bigls ]
			angles_and_defocuses_and_collection_sorted = angles_and_defocuses_and_collection_unsorted.copy()
			angles_and_defocuses_and_collection_sorted.sort()

			anglelines = [ str(a[0]) + "\n" for a in angles_and_defocuses_and_collection_sorted ]
			defocuslines = [ str(d[1]) + "\n" for d in angles_and_defocuses_and_collection_sorted ]

			#c: take a subset of the data matching len(intilts) because the mdoc may have been generated from another mdoc with a different number if images
			writef(anglelines[:len(intilts)], options.path +'/' + stemf_root  + ".rawtlt")
			writef(defocuslines[:len(intilts)], options.path +'/' + stemf_root + ".mdefocus")

			#if '-unsorted' in or 'unsorted' in f:
			#writef(anglelines, options.path +'/' + stemf.replace('.mrc-unsorted','').replace('-unsorted','').replace('unsorted','').replace('.mrc','')  + ".rawtlt")
			#writef(defocuslines, options.path +'/' + stemf.replace('.mrc-unsorted','').replace('-unsorted','').replace('unsorted','').replace('.mrc','') + ".mdefocus")
			#else:
			#	writef(anglelines, options.path +'/' + stemf.replace('.mrc','') + ".rawtlt")
			#	writef(defocuslines, options.path +'/' + stemf.replace('.mrc','') + ".mdefocus")
		
			intiltsfrommdoc = [ d[-1].split('\\')[-1].replace('\n','') + "\n" for d in angles_and_defocuses_and_collection_sorted ]
			tag,newext = comparetilts(intilts,intiltsfrommdoc)

			#intiltsdictl = [ str(int( l.split("[ZValue = " )[-1].split(']\n')[0] ))+ "\t" + str(round(float( l.split("TiltAngle = " )[-1].split('\n')[0] ),2)) +"\n" for l in bigls ]
			#intiltsdict.update( { newindexintiltseries:[ intilt, angle, collectionindex ]} )			

			for i in range(len(angles_and_defocuses_and_collection_sorted)):
				intiltsdict.update( { i:[ os.path.splitext(angles_and_defocuses_and_collection_sorted[i][3])[0].split('\\')[-1]+tag+newext, angles_and_defocuses_and_collection_sorted[i][0], angles_and_defocuses_and_collection_sorted[i][2] ]} )

						
	else:
		angles = getangles( options, len(intilts) )			#This returns angles from -tiltrange to +tiltrange if --negativetiltseries is supplied; from +tiltrange to -tiltrange otherwise
	
		orderedangles = list( angles )
		
		#print "\n(organizetilts) angles are", angles
		
		zeroangle = min( [ math.fabs(a) for a in angles] )		#Find the angle closest to 0 tilt, in the middle of the tiltseries
		indexminangle = None
		try:
			indexminangle = angles.index( zeroangle )			#The middle angle (0 tilt) could have been positive
			zeroangle = angles[ indexminangle ]
		except:
			indexminangle = angles.index( -1*zeroangle )		#Or negative. Either way, we find its index in the list of angles
			zeroangle = angles[ indexminangle ]

	
		angles.sort()
		if not raworder:			#Change angles to go from +tiltrange to -tiltrange if that's the order of the images
			#orderedangles.sort()
			#orderedangles.reverse()
			
			#angles.sort()
			print("raworder is",raworder)
			angles.reverse()
			print("therefore, angles are reversed (largest to smallest)")
			
			#print "However, after reversal, they are", orderedangles
			#print "and angles are", angles
				
		writetlt( angles, options )
			
		if len( intilts ) != len( orderedangles ):
			
			if not options.stackregardless:	
				print("""\n(tomo_tilt_stacker.py)(organizetilts) ERROR: Number of tilt angles 
				and tilt images is not equal.""")
				print("""Number of tilt angles = %d ; number of tilt images = %d """ % ( len( orderedangles ), len( intilts ) ))
				sys.exit()
			else:
				print("""\n(tomo_tilt_stacker.py)(organizetilts) WARNING: Number of tilt angles 
				and tilt images is not equal. Stacking nevertheless since you provided --stackregardless""", options.stackregardless)
				print("""Number of tilt angles = %d ; number of tilt images = %d """ % ( len( orderedangles ), len( intilts ) ))
				
		
		indexesintiltseries = range(len(angles))
		collectionindexes = range(len(angles))
		
		print("indexminangle is", indexminangle)

		if indexminangle != None:
			firstrange = range(0,indexminangle+1)
			secondrange = range(indexminangle+1,len(angles))			
			
			collectionindexes = []
			collectionindexes = list(firstrange) + list(secondrange)
			
			
			firstrange.sort()
			firstrange.reverse()
			
			indexesintiltseries = []
			indexesintiltseries = list(firstrange) + list(secondrange)
		
			
		print("collection indexes are", collectionindexes)
		print("indexes in tiltseries are", indexesintiltseries)
			
			
		for k in range(len(intilts)):
			#print "\nk={}, ntiltangles={}, ncollectionindexes={}, ntilts={}".format(k,len(indexesintiltseries),len(collectionindexes),len(intilts))
			try:
				tiltangle = orderedangles[k]
				#indexintiltseries = angles.index( orderedangles[k] )
			
				indexintiltseries = indexesintiltseries[k]
				collectionindex = collectionindexes[k]
				#print "\nnormal"
			except:
				if options.stackregardless:
					print("\nexception triggered")
					tiltangle = orderedangles[k-1] + orderedangles[k-1] - orderedangles[k-2]
					indexintiltseries = indexesintiltseries[k-1] + indexesintiltseries[k-1] - indexesintiltseries[k-2]
					collectionindex = collectionindexes[k-1] + collectionindexes[k-1] - collectionindexes[k-2]
				
			intiltsdict.update( { indexintiltseries:[ intilts[k],tiltangle,collectionindex ]} )
				#print "\nadded collectionIndex=%d, tiltangle=%f, indexintiltseries=%d" % ( collectionindex, tiltangle, indexintiltseries )
 				
	return intiltsdict


def comparetilts(intilts,intiltsfrommdoc):
	intilts.sort()
	intiltsfrommdoc.sort()
	tags = []
	newext = ''
	print("\n(comparetilts) len(intilts)={}, type(intilts)={}, len(intiltsfrommdoc)={}, type(intiltsfrommdoc={})".format(len(intilts),type(intilts),len(intiltsfrommdoc),type(intiltsfrommdoc)))
	for i in range(len(intilts)):
		stem,ext = os.path.splitext( os.path.basename(intilts[i]) )
		mstem,mext = os.path.splitext( os.path.basename(intiltsfrommdoc[i]) )
		print("\nstem={}, mstem={}".format(stem,mstem))
		
		tag=''
		if len(mstem) < len(stem):
			tag = stem.replace(mstem,'')
		elif len(mstem) > len(stem):
			tag = mstem.replace(stem,'')
	
		if tag:
			tags.append(tag)
		if ext != mext:
			newext = ext

	tags = list(set(tags))
	if tags:
		if len(tags) == 1:
			return tags[0],newext
		elif len(tags) > 1:
			print("\nlen(tags)={}, and tags are={}".format(len(tags),tags))
			print("\nERROR: the tilt images, for example {}, do not match the filenames derived from the .mdoc file; for example {} ".format(intilts[0],intiltsfrommdoc[0]))
			print("\nif the difference between them is the same in all images, this can be fixed, but more than one set of differences was found = {}".format(tags))
			sys.exit(1)
	else:
		return '',newext


def writef(lines,filename):
	with open(filename, 'w') as g:
		g.writelines(lines)
	return


def getindxs( string ):
	
	parsedindxs = set([])
	stringList = list( string.split(',') )
	#print "\nstringList is", stringList
	
	intList=set([])
	for i in range( len(stringList) ):
		print("\n\nstringList[i] is", stringList[i])
		
		if '-' in stringList[i]:
			stringList[i] = stringList[i].split('-')
			
			x1 = int( stringList[i][0] )
			x2 = int( stringList[i][-1] ) + 1
			
			#stringList[i] = [ str( ele ) for ele in xrange( x1, x2 ) ]
			intList.union([ ele for ele in xrange( x1, x2 ) ])
			
		#parsedindxs = parsedindxs.union( set( list( List[i] ) ) )
		
		#intList = set( list(intList) )
	if intList:
		parsedindxs = [ str(ele) for ele in intList ]
	else:
		parsedindxs = stringList
		
	#parsedindxs = list( parsedindxs )
	parsedindxs.sort()
	
	print("\nParsed indexes are", parsedindxs)
	return parsedindxs
	

def makeimglist( inputf, options ):
	
	n = EMUtil.get_image_count( inputf )
	if '.ali' in inputf[-4:] or '.mrc' in inputf[-4:] or '.mrcs' in inputf[-5:] or '.st' in inputf[-3:]:
		n = EMData( inputf,0,True )['nz']

	print("input n is", n)
	print("input is",input)
	
	allindxs = set( [ str(i) for i in range(n) ] )
	finalindxs = set( list( allindxs ) )
	
	print("\nallindxs are", allindxs)
	
	print("\nFinalindxs are", finalindxs)
	
	ints = []
	for id in finalindxs:
		ints.append( int( id ) )
	
	ints.sort()
	ints.reverse()
	
	print("\nfinalindx sorted are", ints)
	#final = []
	#for fi in ints:
	#	final.append( str( fi ) )
	
	lines = []
	for indx in ints:
		lines.append( str(indx) + '\n' )
		
	listfile = options.path + '/list.lst'
	f= open( listfile,'w' )
	f.writelines( lines )
	f.close()
	
	print("\nlistfile is", listfile)
	return listfile


def floatrange(start, stop, step):
	print("\nInside floatrange, start, stop and step are", start, stop, step)
	
	r = start
	kkk=0
	while r <= stop:
		yield r
		#print "r is", r
		r += step
		kkk+=1
		
		if kkk > 180:
			print("ERROR: Something is wrong with your tiltrange, lowesttilt or highesttilt")
			sys.exit()
	
	return


if __name__ == "__main__":

	main()
