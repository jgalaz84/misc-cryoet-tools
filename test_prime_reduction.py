#!/usr/bin/env python
#
# Author: Jesus Galaz-Montoya 11/2020; last modification: 11/2020
#
#

import numpy, sys
import matplotlib.pyplot as plt

def main():
	#c:program to test "empirically" whether prime numbers ever 'reduce to' 3, 6 or 9
	
	ps = primesfrom2to(int(sys.argv[1]))
	print("\nscanning primes from 1 to {}".format(sys.argv[1]))

	np = len(ps)
	
	psr3 = []
	reductions_dict = {}
	reductions_dict_indx = {}
	count = 0
	for p in ps:
		rn = reducenum(p)
		if not rn%3:
			print("\nprime={} reduces to {}".format(p,rn))
			psr3.append(p)
		reductions_dict.update({p:rn})
		reductions_dict_indx.update({count:rn})
		count+=1


	plotdict(reductions_dict,"Digital root for primes","Number","Digital root","primes_dr.png")
	plotdict(reductions_dict_indx,"Digital root for prime indexes","Index","Digital root","primes_dri.png")

	drs_freq_dict = {}
	for i in range(10):
		nval = sum(x == i for x in reductions_dict.values())
		drs_freq_dict.update({i:nval})

	plotdict(drs_freq_dict,"Digital root frequencies for primes","N","Frequency","primes_dr_freq.png")

	npsr3 = len(psr3)
	if npsr3 > 0:
		print('\nfound n={}/{} primes that reduce to 3, 6 or 9'.format(npsr3,np))
		print("\nthey are {}".format(psr3))
	else:
		print("\nno primes out of n={} reduce to 3, 6 or 9".format(np))

	return


def plotdict(dic,t,xl,yl,plotname):
	xs = [a for a in dic.keys()]
	xs.sort()
	ys=[dic[x] for x in xs]

	plt.scatter(xs, ys, color='b', marker='.')

	#plot(xs, ys, color='b', marker='o', linestyle='dashed', linewidth=2, markersize=12)

	plt.title(t)
	plt.xlabel(xl)
	plt.ylabel(yl)

	#plt.axis([min(intensities), max(intensities)])	
	#plt.tick_params(axis='both',which='both',direction='out',length=1,width=2)
	
	#plt.plot(binmids, yfit, 'r--', linewidth=1)

	#plt.grid(True)
	#plt.show()
	
	#if onefile == 'yes':
	plt.savefig(plotname)
	plt.clf()

	return


def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = numpy.ones(n//3 + (n%6==2), dtype=numpy.bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return numpy.r_[((3*numpy.nonzero(sieve)[0][1:]+1)|1)]


def reducenum(num):
	#print("\nprime number to reduce is {}".format(num))

	rnum = sum([int(x) for x in str(num)])
	#print("\ninitial reduction is {}".format(rnum))

	while rnum > 9:
		#print('\nthe number N={} reduced to {}'.format(num,rnum) )
		rnum = sum([int(x) for x in str(rnum)])
	
	print("\nprime number {} reduces to {}".format(num,rnum))
	return rnum


if __name__ == '__main__':
	main()


