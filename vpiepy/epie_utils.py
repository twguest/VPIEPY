

import h5py as h5


import numexpr as ne
import numpy as np

import scipy as sp
import scipy.fftpack as spf


def load_npy_data_cube(filename):
	return np.transpose(np.sqrt(np.load(filename)), (2,0,1))


def load_hdf_data_cube(filename):
	fp = h5.File(filename, 'r')
	data = np.transpose(np.array(np.sqrt(fp.get('/entry/instrument/detector/data'))), (2,0,1))
	fp.close()
	return data


def overlapConstraint( arr1 , arr2 , exit_wave , alpha ):
	
	gamma = alpha / np.max(ne.evaluate("real(abs(arr2)**2)"))
	
	return ne.evaluate("arr1*(1-(gamma*real(abs(arr2)**2)))+(gamma*exit_wave*conj(arr2))")
	#return ne.evaluate("arr1*(1-(gamma*arr1*real(abs(arr2)**2)))+(gamma*exit_wave*conj(arr2))")


def save_ptycholib_csv(filename, array):
	
	pix = array.shape[0]
	
	myfmtstr = '(%+1.18f%+1.18fj) ' * pix
	
	np.savetxt(filename, array, fmt=myfmtstr)

	return 1

def create_stxm_estimate(img_pix, prb_pix, position_vector, dx, energy_vector, fwhm):

	'''
	img_pix : the size of the output image in pixels. output is a square array so if img_pix = 1024 the output will be 1024x1024
	prb_pix : the size of the probe_array used in the ptychography reconstruction
	position_vector : vector of the 2D (x,y) positions corresponding to the scan parameters (in real world units)
	dx : the pixel size of the image (eg, might be something like 500e-9)
	energy_vector : this is a 1D array which corresponds to the total transmitted energy for each diffraction pattern (a measure of the total transmission per probe scan position)
	fwhm : the FWHM of the probe
	'''

	img_mid = np.int( img_pix/2 )
	prb_mid = np.int( prb_pix/2 )

	npts = len( position_vector )
	divpos = (position_vector/dx).astype(np.int)

	fwhm_pix = np.int(fwhm / dx)

	x = np.atleast_2d(np.arange(prb_pix) - prb_mid)
	y = np.atleast_2d(np.arange(prb_pix) - prb_mid)

	radial_array = np.sqrt(x.T**2 + y**2)

	lorentz_array = (1./np.pi) * ((0.5*fwhm_pix)/(radial_array**2 + (0.5*fwhm_pix)**2))

	image_array = np.zeros((img_pix,img_pix)).astype('complex64')

	spx_vector = img_mid + divpos[:,0] - prb_mid
	epx_vector = spx_vector + prb_pix

	spy_vector = img_mid + divpos[:,1] - prb_mid
	epy_vector = spy_vector + prb_pix

	#slicevectorx = slice(spy_vector,epy_vector)
	#slicevectory = slice(spy_vector,epy_vector)

	#slicedatavector = slice(np.arange(npts), np.arange(npts)+1)

	#image_array[ slicevectorx, slicevectory ] += np.abs(energy_vector[slicedatavector] * lorentz_array)*np.exp(complex(0,1)*0.0)

	for i in np.arange( npts ):

		#image_array[ spy:epy, spx:epx ] += np.abs(energy_vector[i] * lorentz_array)*np.exp(complex(0,1)*0.0)
		image_array[ spy_vector[i]:epy_vector[i], spx_vector[i]:epx_vector[i] ] += np.abs((energy_vector[i]) * lorentz_array) + 0.0j#*np.exp(complex(0,1)*0.0)

	return image_array

def broadcast_diffraction(inputarray, padpix, rebpix):
	outputarray = np.zeros((padpix,padpix))
	pm = padpix//2
	rm = rebpix//2
	outputarray[0:rm,0:rm] = inputarray[rm: , rm:]
	outputarray[0:rm,-rm:] = inputarray[rm: ,0:rm]
	outputarray[-rm:,0:rm] = inputarray[0:rm, rm:]
	outputarray[-rm:,-rm:] = inputarray[0:rm,0:rm]
	return outputarray

def modulus_constraint(array1, array2, threshold=1e-9):
	" return array with the phase of array1 but the amplitude of array2. Perform check on the amplitude of array1 to avoid dividing by zero "
	
	return sp.where(np.abs(array1)>threshold, np.abs(array2)*(array1/np.abs(array1)), 0.0)

def overlap_constraint(arr1, arr2, exit_wave, alpha):
	gamma = alpha/np.max(np.abs(arr2)**2)
	return arr1 * (1-(gamma*np.abs(arr2)**2)) + (gamma*exit_wave*np.conj(arr2))

def calculate_error(array1, array2):
	array3 = ne.evaluate("(real(abs(array1))-array2)**2")
	return np.sqrt(np.sum(array3))
