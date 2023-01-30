#!/usr/bin/env python
'''
====================
Author: Jesus Galaz-Montoya - oct/2022, Last update: 13/oct/2022
====================
'''
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

import sys, os

from EMAN2 import *
from EMAN2_utils import *


import time
from datetime import timedelta

def main():
	start = time.perf_counter()

	progname = os.path.basename(sys.argv[0])
	usage = """
	This program calculates the average all vs all distance between ASUs in any given/desired symmetry group
	"""
	
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)

	parser.add_argument("--sym", type=str, default='icos', help="""default=icos.""")
	parser.add_argument("--nophi", action='store_true', default=False, help="""default=False. Use only the first two Euler angles, az and alt.""")

	parser.add_argument("--path", type=str,default='asu_distance',help="""Default=asu_distance. Name of the directory where to store the output results.""")
	parser.add_argument("--ppid", type=int, default=-1, help="Default=-1. Set the PID of the parent process, used for cross platform PPID")

	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n",type=int, default=0, help="verbose level [0-9], higher number means higher level of verboseness.")
	parser.add_argument("--vertexes", action='store_true', default=False,help="""Only works if --sym=icos. This flag will make the program extract only the 12 vertexes from among all 60 symmetry-related units.""") 

	(options, args) = parser.parse_args()

	logger = E2init(sys.argv, options.ppid)

	print("\noptions={}".format(options))

	symletter,symnum = symparse(options)
	symorientations = getsymorientations(options,symletter,symnum)

	n = len(symorientations)

	ds=[]
	for asu1 in range(n):
		for asu2 in range(asu1+1,n):
			d=ptcldistance(options,symorientations[asu1],symorientations[asu2])
			if d:
				ds.append(d)
			else:
				print("\n(main) WARNING: distance between symorientations[asu1]={} and symorientations[asu2]={} is d={}".format(symorientations[asu1],symorientations[asu2],d))

	if options.verbose:
		print("\n(main) number of distances before averaging are len(ds)={}".format(len(ds)))
		print("\n(main) nwhich are d={}".format(ds))

	if ds:
		avg_d = sum(ds)/len(ds)
		print("\n\n\n(main) avg_d={}".format(avg_d))
	else:
		print("\n(main) WARNING: distances are empty, ds={}; something went terribly wrong.".format(ds))

	E2end(logger)

	elapsed = time.perf_counter() - start	
	print("\n(main) run time = "+str(timedelta(seconds=elapsed)))

	return



def symparse(options):
	symnames = ['oct','OCT','icos','ICOS','tet','TET']
	symletter='c'
	symnum=1

	if options.sym:
		if options.verbose:
			print("\n--sym={}".format(options.sym))
			
		if options.sym not in symnames:
			
			symletter = options.sym
			symnum = options.sym
			
			for x in options.sym:
				if x.isalpha():
					symnum = symnum.replace(x,'')
					
			for x in options.sym:
				if x.isdigit():
					symletter = symletter.replace(x,'')
			
			if options.verbose > 8:
				print("\n(symparser) The letter for sym is", symletter)
				print("\n(symparser) The num for sym is", symnum)
		
		if options.sym == 'oct' or options.sym == 'OCT':
			symnum = 8
			
		if options.sym == 'tet' or options.sym == 'TET':
			symnum = 12	
			
		if options.sym == 'icos' or options.sym == 'ICOS':
			symnum = 60	
		
		symnum = int(symnum)
	
		return symletter,symnum
	else:
		print("\n(symparser) ERROR: --sym required")
		sys.exit(1)

	return



def getsymorientations(options,symletter,symnum):
	symorientations = []

	final_ts = []

	t = Transform()
		
	if symnum:
		if options.verbose > 8:
			print("\n(getsymorientations) symnum = {} ".format(symnum))
			print("\n(getsymorientations) while symletter is {}".format(symletter))
		
		if symletter == 'd' or symletter == 'D':
			symnum *= 2
			if options.verbose > 8:
				print("\nsymnum corrected, because symmetry is d; thus, the number of symmetry-related positions is n={}".format(symnum))
			
		#if options.save
		
		for i in range(symnum):
			symorientation = t.get_sym( options.sym , i )
			symorientations.append( symorientation )
		final_ts = symorientations.copy()
	
	if options.sym == 'icos' or options.sym == 'ICOS':
		if options.vertexes:
			if options.verbose > 8:
				print("\n(getsymorientations) fetching vertexes only")
		
			#orientations = genicosvertices ( orientations )

			symorientations = genicosvertexesnew ( symorientations )
			final_ts = symorientations.copy()

	if options.nophi:
		
		ts = []
			
		for o in symorientations:
			rot = symorientation.get_rotation()
			az = rot['az']
			alt = rot['alt']
			phi = rot['phi']
			
			#anglesline = 'az=' + str(az) + 'alt=' + str(alt) + 'phi=' + str(phi) + '\n'
			#anglelines.append(anglesline)
			
			ts.append( Transform({'type':'eman','az':az,'alt':alt}) )

		final_ts = ts.copy()
		print("\n(getsymorientations) transforms without phi are ts={}".format(final_ts))

	if options.verbose:
		print("\n(getsymorientations) generated these many orientations n={}".format(len(final_ts)))
		print("\n(getsymorientations) which are {}".format(final_ts))

	return final_ts


def genicosvertexesnew( syms ):

	newsyms = []
	for s in syms:
		rot = s.get_rotation()
		
		if float(rot['alt']) > 63.0 and float(rot['alt']) < 64.0 and int(round(float(rot['phi']))) == 90:
			
			t = Transform( {'type':'eman','alt':rot['alt'], 'phi':rot['az'], 'az':rot['phi']} )
			newsyms.append( t )
		
		if float(rot['alt']) > 116.0 and float(rot['alt']) < 117.0 and int(round(float(rot['phi']))) ==126:
			t = Transform( {'type':'eman','alt':rot['alt'], 'phi':rot['az'], 'az':rot['phi']} )
			newsyms.append( t )
	
	newsyms.append( Transform({'type':'eman','alt':0.0, 'phi':0.0, 'az':0.0}) )
	
	newsyms.append( Transform({'type':'eman','alt':180.0, 'phi':180.0, 'az':0.0}) )
	
	return( newsyms )


def ptcldistance(options,t1,t2, num=0):
	if options.verbose>19:
		print("\n(ptcldistance) t1={}".format(t1))
		print("\n(ptcldistance) t2={}".format(t2))

	t2inv = t2.inverse()
	if options.verbose>8:
		print("\n(ptcldistance) t2inv={}".format(t2inv))

	product = t2inv * t1
	if options.verbose>8:
		print("\n(ptcldistance) product={}".format(product))

	product_SPIN = product.get_rotation('spin')
	if options.verbose>8:
		print("\n(ptcldistance) product_SPIN={}".format(product_SPIN))

	angular_distance = round(float(product_SPIN["omega"]),2)
	if options.verbose>8:
		print("\n(ptcldistance) angular distance between t1={} and t2={} is d={}".format(t1,t2,angular_distance))
		
	return(angular_distance)


if __name__ == '__main__':
	main()



