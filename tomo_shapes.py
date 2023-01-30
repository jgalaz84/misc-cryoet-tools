#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 04/2020; last modification: 01/2023
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

from past.utils import old_div
from builtins import range

import math
import numpy as np
import random
import os
import sys

from EMAN2 import *
from EMAN2_utils import *
from EMAN2jsondb import JSTask,jsonclasses


def main():
	progname = os.path.basename(sys.argv[0])
	usage = """prog [options]
	This programs can generate 3D shapes and multiple copies of 3D shapes (optionally layered and softened) at different scales and in different orientations, 
	inside a larger 3D volume (conceptually akin to tomograms).
	"""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)

	parser.add_argument("--apix",type=float,default=1.0,help="""Default=1.0. Sampling size of the output tomogram. Any models supplied via --input will be rescaled as needed.""")

	parser.add_argument("--box_size", type=int, default=64, help="""Default=64. Base longest-span for each object created (it can be modified from one object to another by other parameters).""")
	parser.add_argument("--background", type=float, default=None, help="""Default=None. Decimal between 0 and 1 for relative density of background to objects. Protein is ~40 percent denser than water; the default density for objects is 1, yielding ~0.71428571428 for water in comparison.""")

	parser.add_argument("--dilutionfactor", type=int, default=1, help="""Default=1. Alternative that supersedes --nptcls to determine the number of objects in the simulated final volume; 1=total crowdedness, the full volume is occupied with objects; 2=half of the volume is empty; 3=only one third if the volume has objects, etc.""") 
	
	parser.add_argument("--input", type=str, default=None, help="""Default=None. Explicitly list 3D image files to simulate particles inside the 'tomogram'; for example: groel.hdf,tric.hdf,mmcpn.hdf,hiv.hdf""")

	parser.add_argument("--half_shapes", type=str, default=None, help="""Default=None. Takes values "all" or "some". For example, --half_spheres=all. If "all", this will clip the generated shapes in half (for example, sphere's will be transformed into hemishperes, along x, y, or z, randomly assigned); if "half" is provided, only half of the shapes will be clipped in half.""")

	#parser.add_argument("--half_shapes_orthogonal", action="store_true", default=True, help="""Default=True. If on, this will clip the generated shapes in half from x, y, z axis only (one direction per shape, randomly chosen), as opposed to from random directions.""")

	parser.add_argument("--layers", type=int, default=None, help="""Default=None. Max value 4. If --layers=1, the objects will be hollow, if greater than 1, concentric layers with --layers number of "shells" to them (depending on the size of the objects, they may only accomodate 2-3 layers max).""")

	parser.add_argument("--ndiversity", type=int, default=1, help="""Default=1. Number of different shapes to include in simulated volume.""")
	parser.add_argument("--nptcls", type=int, default=None, help="""Default=None. Number of 64x64x64 shapes to include inside final simulated volume. The default is as many particles fit inside the volume v=tomonx*tomony*tomonz.""")

	parser.add_argument("--path", type=str, default='tomo_shapes',help="""Defautl=tomo_shapes. Directory to store results in. The default is a numbered series of directories containing the prefix 'tomo_shapes'; for example, tomo_shapes_02 will be the directory by default if 'tomo_shapes_01' already exists.""")

	parser.add_argument("--ppid", type=int, default=-1, help="""Default=-1. Set the PID of the parent process, used for cross platform PPID""")

	parser.add_argument("--savesteps", action="store_true", default=True, help="""Default=True. If on, intermediate files will be saved to --path.""")
	parser.add_argument("--shapes", type=str, default='all', help="""Default=all. Explicitly list desired shapes, separated by commas; for example: cube,prism,sphere. This overrides --ndiversity. Valid entries: all,cube,sphere,prism,pyramid,cylinder,disc,ellipsoid,cross2,cross3.""")
	parser.add_argument("--softedges", action="store_true", default=False, help="""Default=False. If on, the objects will have a soft instead of a sharp edge.""")

	parser.add_argument("--tomonx", type=int, default=512, help="""Default=512. Size in x for simiulated 'tomogarm'.""")
	parser.add_argument("--tomony", type=int, default=512, help="""Default=512. Size in y for simiulated 'tomogarm'.""")
	parser.add_argument("--tomonz", type=int, default=512, help="""Default=512. Size in z for simiulated 'tomogarm'.""")

	parser.add_argument("--varydensity", action="store_true", default=False, help="""Default=False. If on, different objects will have different density values.""")
	parser.add_argument("--varysize", action="store_true", default=False, help="""Default=False. If on, the objects will be simulated at different scales between 32*32*32 and 64*64*64 volumes.""")
	parser.add_argument("--varyorientation", action="store_true", default=False, help="""Default=False. If on, the objects will be simulated in random orientations.""")

	parser.add_argument("--verbose", "-v", type=int, default=0, help="Default 0. Verbose level [0-9], higner number means higher level of verboseness.")
	
	parser.add_argument("--zc", type=int, default=None, help="""Default=None. This will fix all simulated objects to be centered in the z-plane zc.""")


	(options, args) = parser.parse_args()

	if options.half_shapes and options.half_shapes != 'all' and options.half_shapes != 'some':
		print("ERROR: half_shsapes takes values 'all' or 'some' not {}".format(options.half_shapes))
		sys.exit(1)

	#c:if the objects are layered, the 4th and smallest layer (inward) would be 9x9x9 boxels already for a 64x64x64 object. Can't go much smaller than that.
	if options.layers:
		if options.layers > 4:
			print("\nWARNING: maximum number of layers is 4; changing --layers={} to 4".format(options.layers)) 
			options.layers = 4

	#c:dictionary relating different shape types to integers; note that this is a dictionary of functions, to directly execute them when needed
	#c:there are 7 here, but it's easily expandable as new functions get created

	shapes_dict={'sheet':sheet, 'cube':cube, 'sphere':sphere, 'prism':prism, 'pyramid':pyramid, 'cylinder':cylinder, 'disc':disc, 'ellipsoid':ellipsoid, 'cross2':cross2, 'cross3':cross3, 'eman2_1':1, 'eman2_5':5}
	#shapes_dict={0:cube, 1:sphere, 2:prism, 3:pyramid, 4:cylinder, 5:disc, 6:ellipsoid, 7:cross2, 8:cross3}
	#shapes_dict_str={0:'cube', 1:'sphere', 2:'prism', 3:'pyramid', 4:'cylinder', 5:'disc', 6:'ellipsoid', 7:'cross2', 8:'cross3'}

	#c:make a directory where to store the output and temporary files
	makepath(options,stem='tomo_shapes')

	#c:log each execution of the script to an invisible log file at .eman2log.txt
	logger = E2init(sys.argv, options.ppid)

	#c:create a 'base' volume to be reshaped downstream; i.e., a box with value 1 for all voxels
	base_vol = EMData(options.box_size,options.box_size,options.box_size)
	base_vol.to_one()

	#c:randomly select --ndiversity different shapes from the shapes_dict above
	if options.ndiversity>12:
		options.ndiversity=12

	shape_ids = random.sample(shapes_dict.keys(), options.ndiversity)

	if options.verbose>8:
		print("\n--shapes={}".format(options.shapes))
		#sys.exit(1)
	
	if options.shapes:
		shape_ids=options.shapes.split(',')
		print("\nshape_ids={}".format(shape_ids))
		if 'all' in shape_ids:
			print("\nshapes_dict.keys()={}".format(shapes_dict.keys()))
			shape_ids = shapes_dict.keys()
			print("\nincluding ALL shape_ids={}".format(shape_ids))
		else:
			print("\n 'all' is NOT in shape_ids={}".format(shape_ids))

			

	if options.verbose>8:
		print("\nshape_ids={}".format(shape_ids))

	tomovol = options.tomonx*options.tomony*options.tomonz
	#c:ptclvol is not simply = options.box_size**3
	#c:since the objects can be rotated freely, we imagine them in a box circumscribing the sphere formed by the gyration or rotational average of the object.
	#c:the largest span would be the the hypothenus or diagonal of a options.boxsize^3 volume 
	side=options.box_size
	hyp=round(math.sqrt( 3 * (side**2) ))
	ptclvol=hyp**3
	
	nptclsmax = int( math.ceil(tomovol/ptclvol ) )
	if options.verbose>8:
		print("\nnptclsmax={}".format(nptclsmax))
	
	if options.dilutionfactor:
		print('\n--nptcls was reset from --nptlcs={}'.format(options.nptcls))
		options.nptcls = int(nptclsmax/options.dilutionfactor)
		print('\nto {}'.format(options.nptcls))

	print('\n--nptcls={}, nptclsmax={}'.format(options.nptcls,nptclsmax))
	if options.nptcls > nptclsmax:
		options.nptcls = nptclsmax
		print("\nWARNING: --nptcls is too high. The maximum number of box_size * box_size * box_size objects that fit in volume v={}*{}*{} is nptclsmax={}".format(options.tomonx,options.tomony,options.tomonz,nptclsmax))
	
	#c:generate all the points in a grid where to place each object; i.e., giving each object a box_size x box_size x box_size space within the final volume so that objects don't overlap
	coords=get_coords(options)

	if options.nptcls > len(coords):
		options.nptcls = len(coords)

	#c:generate an equal integer number of instances of each selected shape
	n_per_shape = int( math.ceil( options.nptcls/float(len(shape_ids))))
	if options.verbose:
		print("\ngenerating a total of n={} shapes, and n={} of each of t={} types".format(options.nptcls,n_per_shape,len(shape_ids)))

	#c:randomize the [x,y,x] coordinate elements of 'coords' to avoid placing the same types of objects clustered near each other, in downstream processes
	random.shuffle(coords)
	if options.verbose:
		print("\ngenerated a total of n={} shuffled coordinates".format(len(coords)))
	
	#c:loop over the selected shapes
	ptcl_count = 0 #c:explicitly count the particles since n_per_shape may not exactly divide --nptlcs, and math.ceil most likely will cause for options.nptcls to go over the nptcl limit
	loop_count = 0
	sym = Symmetries.get( 'c1' )
	
	blank_tomo = EMData(options.tomonx,options.tomony,options.tomonz)
	blank_tomo.to_zero()

	output_tomo = blank_tomo.copy()
	if options.background:
		output_tomo.to_one()
		output_tomo *= options.background

	box_size = options.box_size
	
	lines=[]
	print("\nthere are these many shape ids n={}, and they are={}".format(len(shape_ids),shape_ids))
	for j in shape_ids:
		print("\nexamining shape={}".format(j))

		output_label = EMData(options.tomonx,options.tomony,options.tomonz) #c: this will be a "label" in the segmentation sense, meaning a file with objects of the same class
		output_label.to_zero()

		shape = None
		base_vol_c = base_vol.copy() #c:make a fresh copy of the base volume each time we make a new shape type, since many operations downstream are applied 'in place'
		if 'eman2' not in j:
			if options.verbose:
				print("\ngenerating shape of type {}".format(j))
			shape = shapes_dict[j](options,base_vol_c)
			print("\nRETURNING *from* (sheet) fuction, type(shape)={}, shape[minimum]={}, shape[maximum]={}".format( type(shape), shape['minimum'], shape['maximum']) )

			if options.savesteps:
				shape.write_image(options.path+'/shapes_models.hdf',loop_count)
		elif 'eman2' in j:	#c:the 'processors' below binarize the pre-made images to a reasonable threshold, empirically
			if options.verbose:
				print("\nfetching pre-made shape of type {}".format(j))

			shape = test_image_3d(shapes_dict[j]).process("math.fft.resample",{"n":2}).process("threshold.binary",{"value":3.0}).process("mask.auto3d",{"nmaxseed":5,"radius":4})
			if options.savesteps:
				shape.write_image(options.path+'/shapes_models.hdf',loop_count)

		if options.layers:
			shape = gen_layers(options,shape)
			if options.verbose:
				print("\nadded layers to shape {}".format(j))


		if shape != None: #and ptcl_count < options.nptcls and ptcl_count < len(coords):
			print("\nworking with shape={}".format(j))
			print("\nshape[minimum]={}, shape[maximum]={}".format( type(shape), shape['minimum'], shape['maximum']) )
			print("\n\n\n\n\n******* n_per_shape={}".format(n_per_shape))
			
			for k in range(0,n_per_shape):
				print("\n\n\n\n\nFILTEEEEERRRRRRR")
				print("ptcl_count < options.nptcls = {}".format(ptcl_count < options.nptcls ))
				print("ptcl_count < len(coords) = {} ".format(ptcl_count < len(coords) ))

				if ptcl_count < options.nptcls and ptcl_count < len(coords):
					shape_c = shape.copy() #c:for orientations to mean anything (if needed for analyses later), they have to be defined with respect to the same "unrotated" frame of reference, or the unrotated shape

					if options.varydensity:
						intensity=random.uniform(0.9,1.1)
						shape_c*=intensity
						if options.verbose>8:
							print("\nvaried intensity by factor={} for ptcl={} of shape={}, global ptcl #={}".format(intensity,k,j,ptcl_count))
							print("\nVARYDENSITY shape_c[minimum]={}, shape_c[maximum]={}".format( type(shape_c), shape_c['minimum'], shape_c['maximum']) )

					if options.half_shapes:
						decision_factor = 0

						if options.half_shapes == 'all':
							decision_factor = 1
						elif options.half_shapes == 'some':
							decision_factor = k%2

						if decision_factor == 1:
							axis=random.choice(['x','y','z'])
							direction=random.choice([-1,1])
							if axis == 'x':
								if direction == -1:
									shape_c.process_inplace("mask.zeroedge3d",{"x0":box_size/2,"x1":0})
								elif direction == 1:
									shape_c.process_inplace("mask.zeroedge3d",{"x0":0,"x1":box_size/2})
							elif axis == 'y':
								if direction == -1:
									shape_c.process_inplace("mask.zeroedge3d",{"y0":box_size/2,"y1":0})
								elif direction == 1:
									shape_c.process_inplace("mask.zeroedge3d",{"y0":0,"y1":box_size/2})
							elif axis == 'z':
								if direction == -1:
									shape_c.process_inplace("mask.zeroedge3d",{"z0":box_size/2,"z1":0})
								elif direction == 1:
									shape_c.process_inplace("mask.zeroedge3d",{"z0":0,"z1":box_size/2})

					if options.varysize:
						shrink_factor = random.uniform(0.5,1.0)
						shape_c.process_inplace("xform.scale",{"scale":shrink_factor,"clip":box_size})

						if options.verbose>8:
							print("\nvaried size by factor={} for ptcl={} of shape={}, global ptcl #={}".format(shrink_factor,k,j,ptcl_count))
							print("\nVARYSIZE shape_c[minimum]={}, shape_c[maximum]={}".format( type(shape_c), shape_c['minimum'], shape_c['maximum']) )
					
					#c:apply a "transformation" (rotations) to the object only if requested and if the randomly-generated orientation is different from the identity (no rotation)
					orientation = Transform()
					if options.varyorientation and j!='sphere': #shape type 1 is 'sphere', which looks the same in all orientations
						orientation = sym.gen_orientations("rand",{"n":1,"phitoo":1,"inc_mirror":1})[0]
					if orientation != Transform():
						shape_c.transform(orientation)
						
						if options.verbose>8:
							print("\napplied orientation t={} to ptcl={} of shape={}, global ptcl #={}".format(orientation,k,j,ptcl_count))
							print("\nTRANSFORMED shape_c[minimum]={}, shape_c[maximum]={}".format( type(shape_c), shape_c['minimum'], shape_c['maximum']) )

					
					#c:in the future, perhaps save orientations in case they're needed for subtomogram averaging tests later...???
					if options.verbose>8:
						print("\nptcl_count={}, len(coords)={}".format(ptcl_count,len(coords)))
					

					xc,yc,zc = int(coords[ptcl_count][0]),int(coords[ptcl_count][1]),int(coords[ptcl_count][2])

					if options.zc:
						zc = options.zc

					#c:will this cause overlap between objects?
					if options.dilutionfactor > 1:
						xc += random.randint(-options.box_size/2, options.box_size/2)
						yc += random.randint(-options.box_size/2, options.box_size/2)
						if not options.zc:
							zc += random.randint(-options.box_size/2, options.box_size/2)
					
					line = str(j)+'\t'+str(xc)+'\t'+str(yc)+'\t'+str(zc)
					if ptcl_count < options.nptcls - 1:
						line+='\n'
	
					lines.append(line)
					
					print("\n\n\nBEFORE INSERTION type(shape_c)={}, shape_c[minimum]={}, shape_c[maximum]={}".format( type(shape_c), shape_c['minimum'], shape_c['maximum']) )
					
					if options.savesteps:
						shape.write_image(options.path+'/'+j+'_models.hdf',k)
					
					output_label.insert_scaled_sum(shape_c,[xc,yc,zc])

					sys.stdout.flush()

					ptcl_count+=1
				
			
			#c:in case there are different densities across the label, add it to the tomogram before binarization
			output_tomo+=output_label

			#c:for the label to actually be a label it needs to be binarized into a map with 0s (voxels not belonging to the feature) and 1s (voxels belonging to the feature)
			
			#output_label.process_inplace("threshold.binary",{"value":0.01})
			output_label.process_inplace("threshold.binary",{"value":0.0})
			output_label *= -1
			output_label += 1
			output_label_file = options.path+'/label_'+str(j)+'.hdf'
			#output_label.write_image(output_label_file,0,EMUtil.get_image_ext_type("unknown"), False, None, 'int8', not(False))
			output_label.write_image(output_label_file,0)
			print("\nWrote output_label, type(output_label)={}, output_label[minimum]={}, output_label[maximum]={}".format( type(output_label), output_label['minimum'], output_label['maximum']) )


			#if options.verbose:
			print("\nbinarized label for shape={} and saved it to file={}".format(j,output_label_file))

			loop_count+=1

	output_tomo_file=options.path+'/simulated_tomogram.hdf'
	output_tomo_th = output_tomo.process("threshold.belowtominval",{"minval":options.background,"newval":options.background})

	maxval = 1.0
	if options.varydensity:
		maxval = 1.1
	output_tomo_th.process_inplace("threshold.clampminmax",{"maxval":maxval})
	
	output_tomo_th.write_image(output_tomo_file,0)	
	print("\nfinished simulating volume (the 'tomogram') and saved it to file={}".format(output_tomo_file))
	print("\ntype(output_tomo_th)={}, output_tomo_th[minimum]={}, output_tomo_th[maximum]={}".format( type(output_tomo_th), output_tomo_th['minimum'], output_tomo_th['maximum']) )

	
	with open(options.path+'/class_and_coords_file.txt','w') as f:
		f.writelines(lines)

	E2end(logger)
	return


def gen_layers(options,shape):
	
	if options.verbose>8:
		print("\n(gen_layers) start")

	gr=(1+5**0.5)/2 #c:"golden" ratio to be used in calculating the shrinking factor that defines the realtive size between layers

	shape_layered = shape.copy()
	shape_full_size = shape['nx']
	for i in range(options.layers):
		shape_to_shrink=shape.copy()
		shape_shrunk=shape_to_shrink.process('math.fft.resample',{'n':gr**(i+1)})
		#shape_shrunk.process_inplace("threshold.binary",{'value':0.0})
		
		#c:invert the contrast of every other layer
		if i%2 == 0:
	
			shape_shrunk*=-1 

		shape_layered+= clip3d(shape_shrunk,shape_full_size)

	if options.background:
		background_inverse = 1.0/options.background
		shape_layered *= background_inverse



	return shape_layered




def get_coords(options):
	if options.verbose>8: print("\n(get_coords) start")

	#c:since the objects can be rotated freely, the coordinates need to be separated by the hypothenus or diagonal of a cube of side length = options.box_size to prevent overlaps between objects
	side=options.box_size
	
	hyp=round(math.sqrt( 3 * (side**2) )) #This is in 3D: sqrt(nx^2+ny^+nz^2), since nx=ny=nz=side


	xs = mid_points(options.tomonx,hyp,hyp)
	ys = mid_points(options.tomony,hyp,hyp)
	zs = mid_points(options.tomonz,hyp,hyp)

	if not xs or not ys or not zs:
		print("\nERROR: xs and/or ys and/or zs is empty")
		print("\n(get_coords) len(xs)={}\nlen(ys)={}\nlen(zs)={}".format(len(xs),len(ys),len(zs)))
		print("\n(get_coords) xs={}\nys={}\nzs={}".format(xs,ys,zs))
		sys.exit(1)


	return [ [xs[i],ys[j],zs[k]] for i in range(0,len(xs)) for j in range(0,len(ys)) for k in range(0,len(zs)) ]



#Returns the mid points of consecutive sections of size "segment" along the "length" of a line; "step" allows for overlaps
def mid_points(length,segment,step):
	#if options.verbose>8: print("\n(mid_points) start")
	return [int(round(p+segment/2.0)) for p in range(0,length,step) if (p+segment/2.0)<(length-(segment/2.0))]

def cube(options,vol_in):
	if options.verbose>8: print("\n(cube) start; type(vol_in)={}".format(type(vol_in)))
	
	#c:since the cubes might be rotated freely, it needs to be shrunk or masked to a size that will impede any density from going outside a 64^3 box when rotated
	side=vol_in['nx']

	#c:the radius of a circle circumscribed in a box is side/2
	radius=side/2.0

	#the side length of a cube within that circle is:
	new_side=math.sqrt(2*radius**2)
	
	diff=side-new_side

	return vol_in.process("mask.zeroedge3d",{"x0":diff/2.0,"x1":diff/2.0,"y0":diff/2.0,"y1":diff/2.0,"z0":diff/2.0,"z1":diff/2.0})

def sphere(options,vol_in):
	if options.verbose>8: print("\n(sphere) start; type(vol_in)={}".format(type(vol_in)))

	radius=vol_in['nx']/2.0
	return vol_in.process("mask.sharp",{"outer_radius":radius})

def sheet(options,vol_in):


	x0 = int(1)
	x1 = int(1)

	y0 = int(1)
	y1 = int(1)

	z0 = int(options.box_size/2 -1)
	z1 = int(options.box_size/2 -1)
	
	if options.verbose>8: 
		print("\n(sheet), type(vol_in)={}, vol_in[minimum]={}, vol_in[maximum]={}".format( type(vol_in), vol_in['minimum'], vol_in['maximum']) )
		print("\nmasking zeroedge3d with values x0={}, x1={}, y0={}, y1={}, z0={}, z1={}".format(x0,x1,y0,y1,z0,z1) )

	return vol_in.process("mask.zeroedge3d",{"x0":x0,"x1":x1,"y0":y0,"y1":y1,"z0":z0,"z1":z1})
	
def prism(options,vol_in,thickness=None):
	if options.verbose>8: print("\n(prism) start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	thick_to_use=vol_in['nx']/gr
	if thickness!=None:
		thick_to_use=thickness

	to_erase=(vol_in['nx']-thick_to_use)/2.0
	return vol_in.process("mask.zeroedge3d",{"x0":to_erase,"x1":to_erase,"y0":to_erase,"y1":to_erase})


def pyramid(options,vol_in):
	if options.verbose>8: print("\n(pyramid) start; type(vol_in)={}".format(type(vol_in)))

	pyramid_angle = 63.4

	#c:as for the cube, the side-length of the pyramid base sides and its height needs to be scaled so that upon rotation, no densities go outside the box
	old_box=vol_in['nx']
	radius=old_box/2.0
	box=math.sqrt(2*radius**2)

	#c:this is just a trick to use a bigger "negative" box rotated and translated by precise amounts, 4 times, to "carve" each side of a pyramid out of vol_in 
	box_expanded=3*box

	subtractor=vol_in.copy()
	subtractor_expanded = clip3d(subtractor,box_expanded)
	proto_pyramid = subtractor_expanded.copy()

	subtractor_expanded_scaled=subtractor_expanded.process("xform.scale",{"scale":3,"clip":box_expanded})
	subtractor_expanded_scaled_neg=-1*subtractor_expanded_scaled

	tz=2.15*box*math.cos(math.radians(pyramid_angle-90)) #c:This should put the edge face of a slanted negative cube thrice in side length as vol_in to intersetc with the bottom left corner of vol_in 
	print("\ntz={}".format(tz))

	tslant = Transform({"type":"eman","az":0,"alt":pyramid_angle,"phi":0,"tz":tz})
	subtractor_expanded_slanted=subtractor_expanded_scaled_neg.copy()
	subtractor_expanded_slanted.transform(tslant)

	proto_pyramid_steps=[]
	subtractors=[]
	
	for i in range(0,4):
		az=i*90
		t1 = Transform({"type":"eman","az":0,"alt":90})
		t2 = Transform({"type":"eman","az":az,"alt":-90,"phi":0})
		ttot = t2*t1
		subtractor_i = subtractor_expanded_slanted.copy()
		subtractor_i.transform(ttot)
		subtractors.append(subtractor_i)
		proto_pyramid+=subtractor_i
		proto_pyramid_steps.append(proto_pyramid)
		proto_pyramid.process_inplace("threshold.belowtozero",{"minval":0.0})

		sys.stdout.flush()

	pyramid=clip3d(proto_pyramid,box)

	return pyramid

def cylinder(options,vol_in):
	if options.verbose>8: print("\n(cylinder)start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	height=vol_in['nx']
	radius=round(height/gr**2)
	return vol_in.process("testimage.cylinder",{"height":height,"radius":radius})

def disc(options,vol_in):
	if options.verbose>8: print("\n(disc) start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	major=vol_in['nx']/2.0
	minor=int(round(major/gr))
	height=int(round(minor/gr))
	return vol_in.process("testimage.disc",{"major":major,"minor":minor,"height":height})

def ellipsoid(options,vol_in):
	if options.verbose>8: print("\n(ellipsoid) start; type(vol_in)={}".format(type(vol_in)))
	vol_in.to_zero()
	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	a=vol_in['nx']/2.0
	b=round(a/gr)
	c=round(b/gr)
	return vol_in.process("testimage.ellipsoid",{"a":a,"b":b,"c":c,"fill":1})

def cross2(options,vol_in,thickness=None):
	if options.verbose>8: print("\n(cross2) start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	thick_to_use=vol_in['nx']/gr**3
	if thickness!=None:
		thick_to_use=thickness
	
	element1 = prism(options,vol_in,thick_to_use)
	
	element2 = element1.copy()
	t=Transform({"type":"eman","az":0,"alt":90,"phi":0})
	element2.transform(t)

	return element1+element2

def cross3(options,vol_in,thickness=None):
	if options.verbose>8: print("\n(cross3) start; type(vol_in)={}".format(type(vol_in)))

	gr=(1+5**0.5)/2 #c:golden ratio to be used in calculating the shrinking factor that defines the realtive size between layers
	thick_to_use=vol_in['nx']/gr**3
	if thickness!=None:
		thick_to_use=thickness
	
	element1 = prism(options,vol_in,thick_to_use)
	
	element2 = element1.copy()
	t=Transform({"type":"eman","az":0,"alt":90,"phi":0})
	element2.transform(t)

	element3 = element1.copy()
	t=Transform({"type":"eman","az":0,"alt":90,"phi":90})
	element3.transform(t)

	return element1+element2+element3


if __name__ == "__main__":
    main()
    sys.stdout.flush()