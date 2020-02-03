
import numpy as np
import scipy as sp

def randomise_amplitude(pix, minval, maxval):
    absrange = maxval - minval
    return np.random.rand(pix,pix)*absrange + minval

def randomise_phase(pix, minval, maxval):
    phaserange = maxval - minval
    return np.random.rand(pix,pix)*phaserange + minval

def make_super_gaussian(size, fwhm, dx, power, norm=False):
	
	array = np.zeros((size,size))
	radius = sp.hypot( *sp.ogrid[-size//2:size//2,-size//2:size//2] ) * dx

	sigma = fwhm/(2*np.log(2)**(1.0/power))
	
	output = 1.0 * np.exp(-0.5*(radius/sigma)**power)

	if (norm == False):
		return output
	else:
		return output / np.sum(output)

def downstream_prop(inputArray, noshift='true'):
	if noshift=='true':
		return np.fft.fft2(np.fft.fftshift(inputArray), norm='ortho')
	else:
		return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(inputArray), norm='ortho'))

def upstream_prop(inputArray, noshift='true'):
	if noshift=='true':
		return np.fft.fftshift(np.fft.ifft2(inputArray, norm='ortho'))
	else:
		return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(inputArray), norm='ortho'))

def prop_short_distance(array1, propagator, noshift='true'):
	
	'''
	use the angular spectrum method to propagate a complex wavefield a short distance
	very useful when the Fresnel number for the intended propagation is astronomically high
	(i.e. when the Fresnel number makes typical single-FFT based propagation impossible)
	'''

	return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(array1))*propagator))

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

def upsample_array(array1, factor):
	
	pix1 = array1.shape[0]
	mid1 = pix1//2
	
	pix2 = pix1*factor
	mid2 = pix2//2
	
	array2 = np.zeros((pix2,pix2)).astype('complex64')
	
	array2[mid2-mid1:mid2+mid1,mid2-mid1:mid2+mid1] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array1), norm='ortho'))
	
	return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(factor*array2), norm='ortho'))

