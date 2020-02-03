'''

module to calculate the Jones matrices of polarised wavefields

in all cases that follow, we're assuming that the input array represents a complex wavefield of dimensions (pix,pix)
that contains 2 "slices" ... one for X and one for Y

so if we were to initialise the "psi" array it would be as follows,

psi.shape = (2,pix,pix)
psi.dtype = complex64

psi[0,:,:] = the complex 2D wavefield oscillating in the x plane
psi[1,:,:] = the complex 2D wavefield oscillating in the y plane


'''

import numpy as np

J = np.complex(0,1)
PI = np.pi
COS = np.cos
SIN = np.sin
EXP = np.exp

def initialise_vector_wavefield(pix):
    psi_out = np.zeros((2,pix,pix)).astype('complex64')
    psi_out[0] = 1.0 + 0.0*J
    return psi_out

def jones_product(jones_matrix, psi_in):
    """
    multiplies the complex 2d wavefield (representing the x and y plane oscillations) with a 2x2 jones matrix
    that represents some arbitrary birefringent optical element
    
    the output is a complex 2d wavefield (representing the x and y plane oscillations) exiting the optical element
    """
    
    psi_out = np.zeros_like(psi_in).astype(psi_in.dtype)
    
    psi_out[0] = jones_matrix[0,0]*psi_in[0] + jones_matrix[0,1]*psi_in[1]
    psi_out[1] = jones_matrix[1,0]*psi_in[0] + jones_matrix[1,1]*psi_in[1]

    return psi_out

def qwp(theta):
    '''
    returns the Jones matrix for a quarter wave plate whose fast axis is oriented at theta
    '''

    qwp = np.zeros((2,2)).astype('complex64')

    qwp[0,0] = EXP(-J*PI/4.) * (COS(theta)**2 + J*SIN(theta)**2)
    qwp[0,1] = EXP(-J*PI/4.) * ((1-J)*SIN(theta)*COS(theta))
    qwp[1,0] = EXP(-J*PI/4.) * ((1-J)*SIN(theta)*COS(theta))
    qwp[1,1] = EXP(-J*PI/4.) * (SIN(theta)**2 + J*COS(theta)**2)

    return qwp

def hwp(theta):
    '''
    returns the Jones matrix for a half wave plate whose fast axis is oriented at theta
    '''
    
    hwp = np.zeros((2,2)).astype('complex64')
    
    hwp[0,0] = EXP(-J*PI/2.) * (COS(theta)**2 - SIN(theta)**2)
    hwp[0,1] = EXP(-J*PI/2.) * 2.0*COS(theta)*SIN(theta)
    hwp[1,0] = EXP(-J*PI/2.) * 2.0*COS(theta)*SIN(theta)
    hwp[1,1] = EXP(-J*PI/2.) * (SIN(theta)**2 - COS(theta)**2)

    return hwp

def lin_pol(theta):
    '''
    returns the Jones matrix for a linear polariser whose axis is oriented at theta
    '''	
    
    lin_pol = np.zeros((2,2)).astype('complex64')

    lin_pol[0,0] = COS(theta)*COS(theta) + 0*J
    lin_pol[0,1] = COS(theta)*SIN(theta) + 0*J
    lin_pol[1,0] = COS(theta)*SIN(theta) + 0*J
    lin_pol[1,1] = SIN(theta)*SIN(theta) + 0*J

    return lin_pol

def prop_through_qwp(psi_in, theta):
    return jones_product(qwp(theta), psi_in)

def prop_through_hwp(psi_in, theta):
    return jones_product(hwp(theta), psi_in)

def prop_through_lin_pol(psi_in, theta):
    return jones_product(lin_pol(theta), psi_in)


def rotate(psi_in, theta):
    
    psi_out = np.zeros_like(psi_in).astype( psi_in.dtype )
    
    psi_out[0,:,:] = np.cos(theta) * psi_in[0] + np.sin(theta) * psi_in[1]
    psi_out[1,:,:] = -np.sin(theta) * psi_in[0] + np.cos(theta) * psi_in[1]
    
    return psi_out

def rotate_exactly(psi_in, theta):
    
    psi_out = np.zeros_like(psi_in).astype(psi_in.dtype)
    psi_out = np.cos(theta)*psi_in[0] + np.sin(theta)*psi_in[1]
    
    return psi_out

def rotate_coords(psi_in, theta):
    """
    Rotate the basis coordinates by theta (CCW in radians)

    Arguments:
        psi_in {2 component complex 2D array} -- [description]
        theta {angle in radians} -- [description]
    """
    
    psi_out = np.zeros_like(psi_in)

    psi_out[0] = np.cos(theta)*psi_in[0] + np.sin(theta)*psi_in[1]
    psi_out[1] = -np.sin(theta)*psi_in[0] + np.cos(theta)*psi_in[1]

    return psi_out

def rotate_vector(psi_in, theta):
    """
    Rotate an input vector in (x,y) coords by theta (CCW in radians) to a new point in (x,y)
    Once, again, this function rotates the VECTOR in (x,y) to a NEW POINT in (x,y)
    
    Arguments:
        psi_in {[type]} -- [description]
        theta {[type]} -- [description]
    """
    psi_out = np.zeros_like(psi_in)

    psi_out[0] = np.cos(theta)*psi_in[0] - np.sin(theta)*psi_in[1]
    psi_out[1] = np.sin(theta)*psi_in[0] + np.cos(theta)*psi_in[1]

    return psi_out