#!/usr/bin/env python
#====================
#Author: Jesus Galaz-Montoya Dec/2021 , Last update: December/05/2021
#====================
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
from builtins import range
from EMAN2_utils import *
from EMAN2 import *
import math

from sys import argv
import time
from datetime import timedelta


def main():
	
	start = time.perf_counter()

	usage = """e2spt_gen_file_list.py <options> . 
	Small script to generate a conical mask. Potential use in better isolating/masking trimeric spikes from enveloped virions, anchored to the membrane via thin stalks.
	"""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)	

	parser.add_argument("--angle", type=float, default=None, help="Default=None. Angle for the conical mask. Ignored if --height and --base are provided simultaneously")
	parser.add_argument("--apix", type=float, default=None, help="Default=None. Set sampling size")

	parser.add_argument("--base", type=float, default=None, help="Default=None. Base for the conical mask.")

	parser.add_argument("--clip", type=int, default=None, help="Default=usually 2x the largest of the provided or calculated --base and --height. Box size to clip the final mask to if the default seems too large. May take trial and error to avoid cutting out parts of the mask.")

	parser.add_argument("--flip", action="store_true", default=False, help="Default=False. Have the tip of the cone point downwards rather than upwards.")
	
	parser.add_argument("--height",type=float,default=None, help="""Default=None. Height for the conical mask.""")

	parser.add_argument("--lowpass", type=float, default=None, help="Default=None. Resolution in Å to lowpass to.")

	parser.add_argument("--mrcoutput", action="store_true", default=False, help="Default=False. If on, this will write .mrc files instead of .hdf files.")

	parser.add_argument("--ppid", type=int, default=-1, help="Default=-1. Set the PID of the parent process, used for cross platform PPID")
	
	parser.add_argument("--savesteps", action="store_true", default=False, help="Default=False. If on, this will write intermedite files.")

	parser.add_argument("--verbose", "-v", type=int, dest="verbose", action="store", metavar="n", default=0, help="verbose level [0-9], higner number means higher level of verboseness")

	(options, args) = parser.parse_args()	
	
	print("\nLogging")
	logger = E2init(sys.argv, options.ppid)

	#c:figure out which parameters need to be calculated from the given ones
	halfbase, angle, height = None, None, None

	if options.angle:
		angle = options.angle
		if options.base and not options.height:
			halfbase = options.base/2.0
			height = math.fabs( halfbase * math.tan( math.degrees(options.angle) ) )
			if options.verbose:
				print("\ndetermined height={}".format(height))
		elif options.height and not options.base:
			halfbase = math.fabs( options.height / math.tan(math.degrees(options.angle)) )
			height = options.height
			if options.verbose:
				print("\ndetermined halfbase={}".format(halfbase))
		elif options.base and options.height:
			options, halfbase, angle, height = baseandheight(options)
			if options.verbose:
				print("\noverwriting --angle with determined angle={}".format(angle))
	else:
		if options.base and options.height:
			options, halfbase, angle, height = baseandheight(options)
			if options.verbose:
				print("\ndetermined angle={}".format(angle))
		else:
			print("\n(main) ERROR: in the absence of --angle, which currently is angle={}, both --base and --height are required.".format(options.angle))
			sys.exit(1)

	base =  halfbase*2
	
	#c:box with generous padding because of the weird way in which the cone is created, space is needed to manipulate the densities (there's probably a much better way to do this...)
	box = int(max(height,base)*2)

	a =EMData(box,box,box)
	a.to_one()
	xmask=(box/2)-1

	apix=1.0
	if options.apix and options.apix > 1.0:
		apix=options.apix
		a['apix_x'] = apix
		a['apix_y'] = apix
		a['apix_z'] = apix

	#c:make a thin slice of 2 pixel width inside the 3d box
	aslice = a.process("mask.zeroedge3d",{"x0":xmask,"x1":xmask})
	if options.savesteps:
		aslice.write_image("slice.hdf",0)

	#c:pad the box so that no densities will go outside when rotated
	hypbox = math.sqrt(2*math.pow(box/2,2))
	aslicehyp = clip3d(aslice,hypbox)

	#c:apply a translation to get the slice off-center to be able to generate a volume by applying rotational symmetry later
	x = -1*(box/2.0 - halfbase)
	rot = 90-angle

	t = Transform({"type":"eman","alt":rot,"tz":x})
	aslice_t = aslicehyp.copy()
	aslice_t.transform(t)
	if options.savesteps:
		aslice_t.write_image("slice_rot_and_trans.hdf",0)

	#c:clip the box back to the original size and mask from the edges in x and y to make a triangle
	aslice_t_box = clip3d(aslice_t,box)

	zmask = box/2
	ymask = box/2
    
	b=EMData(box,box,box)
	b.to_one()

	bmx=b.process("mask.zeroedge3d",{"y0":box/4,"y1":box/4,"z0":box/4,"z1":box/4})
	bmx*=-1

	tx=Transform({"type":"eman","ty":-1*box/4})
	bmxt=bmx.copy()
	bmxt.transform(tx)

	bmy=b.process("mask.zeroedge3d",{"x0":box/4,"x1":box/4,"z0":box/4,"z1":box/4})
	bmy*=-1

	ty=Transform({"type":"eman","tz":-1*box/4})
	bmyt=bmy.copy()
	bmyt.transform(ty)

	atriangle = aslice_t_box + bmyt + bmxt
	atriangle.process_inplace("threshold.belowtozero",{"minval":0.0})

	if options.verbose > 9:
		check_image_health(atriangle,'triangle')

	if options.savesteps:
		savetmp(options,atriangle,"triangle.hdf")

	arot=atriangle.copy()
	trot=Transform({"type":"eman","alt":90})
	arot.transform(trot)

	if options.verbose > 9:
		check_image_health(arot,'triangle_rot')

	if options.savesteps:
		savetmp(options,arot,"triangle_rot.hdf")

	asymr = arot.process("xform.applysym",{"sym":"c100"})

	freq=1/(4*apix)
	if options.lowpass:
		freq = 1.0/options.lowpass
	
	asymr.process_inplace("threshold.binary",{"value":0.00001})
	asymr.process_inplace("filter.lowpass.gauss",{"cutoff_freq":freq})

	prj = asymr.project("standard",Transform())
	prjnx = prj['nx']
	prjny = prj['ny']
	prj.process_inplace("threshold.binary",{"value":0.001})
	ones=0
	j=int(prjny/2)
	for i in range(prjnx):
		val=prj.get_value_at(i,j)
		if val>0.0:
			ones+=1

	scale=1.0
	if ones != base:
		scale=base/ones
		print("\nscaling by factor={} because ones={} and options.base={}".format(scale,ones,base))
		asymr.process_inplace("xform.scale",{"scale":scale,"clip":prjnx})
		asymr.process_inplace("xform.centerofmass")
		asymr['apix_x']=apix
		asymr['apix_y']=apix
		asymr['apix_z']=apix

	if options.clip:
		asymr.process_inplace("xform.centerofmass")
		asymr = clip3d(asymr,options.clip)

	outfile = "cone_h"+str(int(round(height))) + "b" + str(int(round(base))) +".hdf"
	if options.mrcoutput:
		outfile=outfile.replace('.hdf','.mrc')

	asymr.write_image(outfile,0)

	elapsed = time.perf_counter() - start
	
	print("\ntotal run time:")
	print(str(timedelta(seconds=elapsed)))

	E2end(logger)

	return

def baseandheight(options):
	options.angle = None
	if options.verbose:
		print("\nWARNING: angle being reset to None since --base and --height suffice.")
	halfbase = options.base/2.0
	angle = math.fabs( math.degrees(math.atan(options.height/(halfbase))))
	height = options.height
	print("\ndetermined halfbase={} angle={}".format(halfbase,angle,height))

	return options, halfbase, angle, height


def check_image_health(img,tag='img'):

	tmin = img['minimum']
	tmax = img['maximum']
	tmean = img['mean']
	tstd = img['sigma']

	print("\n{} min={} max={} mean={} std={}".format(tag,tmin,tmax,tmean,tstd))
	return


def savetmp(options,img,f):
	if options.mrcoutput:
		outtriangle=f.replace('.hdf','.mrc')
	img.write_image(f,0)
	if options.verbose:
		print("\nsaved tmp file {}".format(f))

	return


if '__main__' == __name__:
	main()
