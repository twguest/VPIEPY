

'''
in order for multislice to work, we need to have the fully recovered
wavefields (the probe wavefield and the object transmission function)
WITHOUT the Fresnel propagators embedded in them

so we have to somehow,
1/ propagate the wavefield from detector to sample
2/ remove the Fresnel propagator terms
3/ apply the inverse multi-slice propagation
4/ apply the ptychography overlap constraints at each slice
5/ recalculate the forward pass until the final slice
6/ re-apply the Fresnel propagator terms
7/ propagate to the detector plane
8/ repeat
'''



import numpy as np

import scipy as sp
import pylab as pyl

import pdb

import os

import scipy.fftpack as spf

import timeit
import glob

import epie_utils as epie
import optics_utils as optics

def overlap_constraint( arr1 , arr2 , exit_wave , alpha ):
	
	gamma = alpha / np.max( np.abs( arr2 )**2 )
	
	return arr1 * ( 1 - ( gamma * np.abs( arr2 )**2 ) ) + ( gamma * exit_wave * np.conj( arr2 ) )
	


scanpath = r'C:\Users\daemo\OneDrive - LA TROBE UNIVERSITY\code\data_for_trey\sim_data\henry_02\\'

if ( pyl.isinteractive() == False ):
	print("turning pylab interactive mode on")
	pyl.ion()
else:
	print("pylab interactive mode already on")



J = complex(0,1)
PI = np.pi

iterations = 1000

update_object_after = 1
update_probe_after = 10

watchprogress = 1
watch_dc = 10


# ... load_det_probe and load_sam_probe CANNOT both be 1
# ... they CAN both be zero but only one of them can be 1
#load_det_probe = 0
#load_sam_probe = 1

load_probe = 0

load_object_estimate = 1

load_position_correction_file = 0

position_correction = 0
position_correction_update_after = 10
position_correction_dc = 30
pcount = 0

update_whitefield_amp = 0


alpha_object = 1.0
alpha_probe = 0.5

cflip = 0

amp_phase_object = 0
amp_phase_object_update_after = 30
amp_phase_object_dc = 10

sub_incoh = 0
calc_incoh_after = 10
sub_incoh_after = 20

geometry_array = np.genfromtxt( scanpath + 'fcdi_config.txt' , dtype = None )

wl = np.float( geometry_array[0] )
N = np.int( np.float( geometry_array[ 1 ] ) )
dxd = np.float( geometry_array[ 2 ] )

rebfac = np.int( np.float( geometry_array[ 3 ] ) )
padfac = np.int( np.float( geometry_array[ 4 ] ) )
fovfac = np.int( np.float( geometry_array[ 5 ] ) )

rebpix = N//rebfac
rebmid = rebpix//2
rebqrt = rebmid//2

padpix = rebpix*padfac
padmid = padpix//2
padqrt = padmid//2

fovpix = padpix*fovfac
fovmid = fovpix//2
fovqrt = fovmid//2

nprbx = np.int(geometry_array[6])
nprby = np.int(geometry_array[7])

nslices = np.int(geometry_array[8])
object_thickness = np.float(geometry_array[9])
dslice = object_thickness/nslices

z01 = np.float(geometry_array[10])
z12 = np.float(geometry_array[11])
z23 = np.float(geometry_array[12])
z24 = np.float(geometry_array[13])
z43 = z23 - z24
z34 = -z43

dx3 = dxd*rebfac
dx2 = (wl * z23) / (padpix * dx3)
dx1 = (wl * z12) / (padpix * dx2)
dx4 = (wl * z43) / (padpix * dx3)

bda_diam = np.float(geometry_array[14])
bda_rad = bda_diam/2

gridpath = geometry_array[15].decode('UTF-8')
datapath = geometry_array[16].decode('UTF-8')

fulldatapath = scanpath + gridpath + datapath + f'_r{rebpix:04}/'


spec_data = sp.loadtxt( scanpath + gridpath + 'smarpodpositions.txt', delimiter=',' )

ptych_num = spec_data.shape[0]


xscansize = np.max(spec_data[:,0]) - np.min(spec_data[:,0])
yscansize = np.max(spec_data[:,1]) - np.min(spec_data[:,1])

midposx = np.min(spec_data[:,0]) + xscansize/2.0
midposy = np.min(spec_data[:,1]) + yscansize/2.0


# centre the scan and convert the positions to pixel units
pos_x = ( spec_data[:,0] - midposx ) // dx4
pos_y = ( spec_data[:,1] - midposy ) // dx4

radpix = sp.hypot(*sp.ogrid[-padpix/2.:padpix/2.,-padpix/2.:padpix/2])


dataFilename = f'pos_000000_to_{ptych_num-1:06}_int_sum.npy'

data = np.load(fulldatapath + dataFilename)

if data.shape[0] == data.shape[1]:
	data = data.transpose(2,0,1)

data_energy = np.sum(data, axis=(1,2))                                                                
inverse_data_energy = 1.0 / data_energy

data = np.sqrt(data)


prefactor = PI*wl*dslice
dk4 = 1.0/(padpix*dx4)

dnstreamAngSpecProp = np.fft.fftshift(np.exp(J * (-1.0*prefactor) * (radpix*dk4)**2))
upstreamAngSpecProp = np.fft.fftshift(np.exp(J * (+1.0*prefactor) * (radpix*dk4)**2))

# simulate an initial estimate of the probe just in case we can't load a previous one
psi_pup_est = sp.where(radpix*dx1<bda_rad, 1.0, 0.0).astype('complex64')
psi_foc_est = optics.downstream_prop(psi_pup_est, noshift='False')
phase_z123 = (PI/wl)*((1/z12)+(1/z23))*(radpix*dx2)**2
psi_det_est = optics.downstream_prop(psi_foc_est*np.exp(J*phase_z123), noshift='False')
phase_z234 = (PI/wl)*((1/z23)+(1/z34))*(radpix*dx3)**2
psi_sam_est = optics.upstream_prop(psi_det_est*np.exp(J*phase_z234), noshift='False')

# load probe arrays
probe_cube_fname  = f'probe_guess_nprbx{nprbx:03d}_nprby{nprby:03d}_r{rebpix:04d}_p{padpix:04d}.npy'
probe_guess_fname = f'probe_guess_r{rebpix:04d}_p{padpix:04}.npy'
probe_det_fname   = f'psi_det_est_r{rebpix:04d}_p{padpix:04d}.npy'

if load_probe == 1:
	if os.path.exists(scanpath+gridpath+probe_cube_fname):
		print("loading probe_cube file ... ")
		probes = np.load(scanpath+gridpath+probe_cube_fname)
		probe_energy = np.max( np.sum(np.abs(probes)**2, axis=(2,3)) )
		probes *= np.sqrt(np.max(data_energy)/probe_energy)
	else:
		if os.path.exists(probe_guess_fname):
			print("probe_cube file not found! ... loading probe_guess file")
			new_probe = np.load(scanpath+gridpath+probe_guess_fname)
		elif os.path.exists(probe_det_fname):
			print("neither probe_cube nor probe_guess files found! ... loading psi_det_est file")
			new_probe = np.load(scanpath+gridpath+probe_guess_fname)
			new_probe = optics.upstream_prop(probe_det_est * np.exp(complex(0,1)*phase_z234))
		else:
			print("no probe files found ... using simulated probe")
			new_probe = np.copy(psi_sam_est)
			probe_energy = np.sum(np.abs(new_probe)**2)
			new_probe *= np.sqrt(np.max(data_energy)/probe_energy)
			probes = np.repeat(new_probe[np.newaxis,:,:], nprbx, axis=0)
			probes = np.repeat(probes[np.newaxis,:,:], nprby, axis=0)
else:
	new_probe = np.copy(psi_sam_est)
	probe_energy = np.sum(np.abs(new_probe)**2)
	new_probe *= np.sqrt(np.max(data_energy)/probe_energy)
	probes = np.repeat(new_probe[np.newaxis,:,:], nprbx, axis=0)
	probes = np.repeat(probes[np.newaxis,:,:], nprby, axis=0)


probe_guess = np.copy(probes[nprby//2, nprbx//2])
probe_guess = np.repeat(probe_guess[np.newaxis,:,:], nslices, axis=0)
new_probe = np.copy(probe_guess)

object_guess_fname0 = f'object_guess_nprbx{nprbx:03d}_nprby{nprby:03d}_nslc{nslices:03d}_r{rebpix:04}_p{padpix:04}_f{fovpix:04}.npy'
object_guess_fname1 = f'object_guess_r{rebpix:04}_p{padpix:04}_f{fovpix:04}.npy'

if load_object_estimate == 1:
	if os.path.exists(scanpath+gridpath+object_guess_fname0):
		print("loading object multi-probe estimate ... ")
		object_guess = np.load(scanpath + gridpath + object_guess_fname0)
	elif os.path.exists(scanpath+gridpath+object_guess_fname1):
		print("object multi-probe estimate file not found! ... loading object single probe estimate")
		object_guess = np.load(scanpath + gridpath + object_guess_fname1)
		object_guess = np.repeat(object_guess[np.newaxis,:,:], nslices, axis=0)
	else:
		print("no object files found... initialising random object")
		rand_amp = 0.9 + np.random.rand(fovpix,fovpix)/10.
		rand_phase = (np.pi/2.)*np.random.rand(fovpix,fovpix) - np.pi/4
		object_guess = rand_amp * np.exp(J*rand_phase)
		object_guess = np.repeat(object_guess[np.newaxis,:,:], nslices, axis=0)
else:
	rand_amp = 0.9 + np.random.rand(fovpix,fovpix)/10.
	rand_phase = (np.pi/2.)*np.random.rand(fovpix,fovpix) - np.pi/4
	object_guess = rand_amp * np.exp(J*rand_phase)
	object_guess = np.repeat(object_guess[np.newaxis,:,:], nslices, axis=0)
	

fftshiftmask = np.zeros((padpix,padpix))
fftshiftmask[padmid-rebmid:padmid+rebmid,padmid-rebmid:padmid+rebmid] = 1.0


rx = fovmid - padmid
ry = fovmid - padmid


error = np.zeros((ptych_num, iterations))

scaledxpos = np.copy(spec_data[:,0] - midposx)
scaledxpos = (scaledxpos-scaledxpos.min())/((2.0*scaledxpos.max())+1e-9)

scaledypos = np.copy(spec_data[:,1] - midposy)
scaledypos = (scaledypos-scaledypos.min())/((2.0*scaledypos.max())+1e-9)

probe_energy = np.sum(np.abs(new_probe)**2)

new_probe *= np.sqrt( np.max(data_energy)/probe_energy )

timea = timeit.time.time()

diffdata = np.zeros((padpix,padpix))

# declare an "incoherent" array which converges to contain an estimate of the incoherent scattered INTENSITY at the detector
runincoh = np.ones((1, padpix,padpix)).astype('float32')
tempincoh = np.ones((1, padpix,padpix)).astype('float32')

runincohdenom = 0
incohenergy = np.zeros(iterations)

print(" Reconstruction begins now...")
pdb.set_trace()
print(" Reconstruction begins now...")

probe_guess = np.copy(new_probe)
esw = np.zeros((nslices,padpix,padpix)).astype('complex64')
esw_energy = np.zeros(3)


for i in np.arange( iterations ):
	
	# at the start of every iteration, jumble up the order of projections
	rand_pos = np.random.permutation( ptych_num )
	
	for j in np.arange( ptych_num ):
		
		# randomly select the projection to use
		k = rand_pos[ j ]

		prbxidx = np.int(np.floor(nprbx*scaledxpos[k]))
		prbyidx = np.int(np.floor(nprby*scaledypos[k]))
		
		# work out the ROI on the larger, object array that corresponds to the k^th projection
		spx = rx + np.int(pos_x[k])
		epx = spx + padpix
		
		spy = ry + np.int(pos_y[k])
		epy = spy + padpix
		
		# 'cut' out that part of the larger, object array
		obj1 = object_guess[:,spy:epy,spx:epx]
		
		diffdata = epie.broadcast_diffraction(data[k], padpix, rebpix)
		
		diffengy = np.copy(inverse_data_energy[k])
		
		probe_guess[0] = np.copy(probes[prbyidx, prbxidx])
		
		esw[0] = probe_guess[0] * obj1[0]
		
		for l in range(nslices-1):	
			# propagate to the next slice
			probe_guess[l+1] = optics.prop_short_distance(esw[l], dnstreamAngSpecProp)
			# multiply by the probe
			esw[l+1] = probe_guess[l+1] * obj1[l+1]

		esw_energy = np.sum(np.abs(esw)**2, axis=(1,2))
		
		# PROPAGATE TO THE DETECTOR PLANE
		dpw = optics.downstream_prop(esw[nslices-1])
		
		# APPLY MODULUS CONSTRAINT
		new_dpw = optics.modulus_constraint(dpw, diffdata, 1e-3 )
		
		# PROPAGATE BACK TO SAMPLE PLANE
		new_esw = optics.upstream_prop(new_dpw)

		new_esw_energy = np.sum(np.abs(new_esw)**2)
	
		if i > update_probe_after:
			new_probe[nslices-1] = overlap_constraint(probe_guess[nslices-1], obj1[nslices-1], new_esw, alpha_probe)

		if i > update_object_after:
			#new_obj1[nslices-1] = overlap_constraint(obj1[nslices-1], probe_guess[nslices-1], new_esw, alpha_object)
			object_guess[nslices-1,spy:epy,spx:epx] = overlap_constraint(obj1[nslices-1], probe_guess[nslices-1], new_esw, alpha_object)
		
		for l in reversed(range(nslices-1)):
			# propagate to the previous slice
			new_esw = optics.prop_short_distance(new_probe[l+1], upstreamAngSpecProp)
			
			if i > update_probe_after:
				new_probe[l] = overlap_constraint(probe_guess[l], obj1[l], new_esw, alpha_probe)
			
			if i > update_object_after:
				object_guess[l,spy:epy,spx:epx] = overlap_constraint(obj1[l], probe_guess[l], new_esw, alpha_object )
		
		probes[prbyidx, prbxidx] = np.copy(new_probe[0])

		print( i, j)
			
	#print( i )


'''
phase_factor = sp.exp( complex( 0 , 1 ) * (np.pi/wl) * ((z12**-1.)+(z23**-1.))*radpix)

fresnelPreFac_34 = np.exp( ((J*PI)/(wl*z34)) * (radpix*dx4)**2 )
fresnelPosFac_34 = np.exp( (J*PI/wl) * ( z23**-1. + z34**-1. ) * ( radpix * dx3 )**2 )
fresnelPosFac_34 = np.exp( complex( 0 , 1 ) * ( np.pi / wl ) * ( z23**-1. + z34**-1. ) * ( radpix * dx3 )**2 )

fresnelPreFac_43 = np.exp( complex( 0 , 1 ) * ( np.pi / wl ) * ( z43**-1. - z23**-1. ) * ( radpix * dx3 )**2 )
fresnelPosFac_43 = np.exp( complex( 0 , 1 ) * ( np.pi / ( wl * z43 ) ) * ( radpix * dx4 )**2 )
'''
