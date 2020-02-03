'''


iterative ptychographical reconstruction assuming vector wavefields


the algorithm is developed using the theory outlined in,

"Ptychography in anisotropic media, Ferrand, Allain, Chamard, 2015"


'''

import jones_matrices as jones

import numexpr as ne
import numpy as np


import optics_utils as optics


import pylab as pyl

import scipy as sp

from tqdm import tqdm
from matplotlib import pyplot as plt


pi = np.pi
class vPIE:
    
    def __init__(self, scan_path, 
                 pmodes = 3, amodes = 3,
                 iterations = 5, nruns = 1, ptych_num = 360,
                 WL = 660e-09,
                 bit_precision = 32,
                 vpie_config_num = 1):
        
        self.bit_precision = bit_precision
        self.r_dtype = 'float{}'.format(bit_precision)
        self.c_dtype = 'complex{}'.format(bit_precision*2)

        self.J = complex(0,1)
        self.PI = np.pi
        
        self.pmodes = pmodes
        self.amodes = amodes
        self.nruns = nruns
        self.iterations = iterations

        self.scan_path = scan_path 
        
        self.WL = 660e-09
        self.K = 2.0*self.PI/self.WL
        
        self.vpie_config = np.loadtxt( scan_path + "vpie_config_%02d.txt" % vpie_config_num )
        self.probe_path = scan_path
        
        self.detpix = 2048
        self.dxd = 6.5e-6

        self.rebfac = 8
        self.padfac = 1
        self.fovfac = 4
        
        self.rebpix = self.detpix//self.rebfac
        self.rebmid = self.rebpix//2
        self.rebqrt = self.rebmid//2
        
        self.padpix = self.rebpix * self.padfac
        self.padmid = self.padpix//2
        self.padqrt = self.padmid//2
        
        self.sampix = self.padpix * self.fovfac
        self.sammid = self.sampix//2
        self.samqrt = self.sammid//2
        
        self.dx3 = self.rebfac * self.dxd
        
        self.z12 = 0.25
        self.z23 = 0.30
        self.z24 = 0.04
        
        self.z43 = self.z23 - self.z24
        self.z34 = -self.z43
        
        self.dx2 = (self.WL*self.z23 ) / (self.padpix * self.dx3)
        self.dx1 = (self.WL*self.z12 ) / (self.padpix * self.dx2)
        self.dx4 = (self.WL*self.z43 ) / (self.padpix * self.dx3)
        
        
        
        self.theta_p = np.zeros( self.pmodes )
        self.theta_a = np.zeros( self.amodes )
        
        self.ptych_num = ptych_num
        self.npts = ptych_num
        
        self.spx = self.padmid - self.rebmid
        self.epx = self.padmid + self.rebmid
        
        
        self.trans = np.ones(((2,2,self.sampix,self.sampix))).astype(self.c_dtype)
        self.trans[0,0] = optics.randomise_amplitude(self.sampix,0.8,1.0) * np.exp(self.J*optics.randomise_phase(self.sampix,-self.PI,self.PI))
        self.trans[0,1] = optics.randomise_amplitude(self.sampix,0.8,1.0) * np.exp(self.J*optics.randomise_phase(self.sampix,-self.PI,self.PI))
        self.trans[1,0] = optics.randomise_amplitude(self.sampix,0.8,1.0) * np.exp(self.J*optics.randomise_phase(self.sampix,-self.PI,self.PI))
        self.trans[1,1] = optics.randomise_amplitude(self.sampix,0.8,1.0) * np.exp(self.J*optics.randomise_phase(self.sampix,-self.PI,self.PI))
        
        self.psi_analyser_est = np.zeros((self.pmodes, self.amodes, self.padpix, self.padpix)).astype(self.c_dtype)
        self.cplx_diff = np.zeros((self.pmodes, self.amodes, self.padpix, self.padpix)).astype(self.c_dtype)
        
        self.arr_A = np.zeros((self.pmodes, self.padpix, self.padpix)).astype(self.c_dtype)
        self.arr_B = np.zeros((self.pmodes, self.padpix, self.padpix)).astype(self.c_dtype)
        
        self.psi_det_est = np.zeros((self.pmodes, self.amodes, self.padpix, self.padpix)).astype(self.c_dtype)
        
        self.trans_crop = np.zeros((2, 2, self.padpix, self.padpix)).astype(self.c_dtype)
        
        
        self.trans = np.load(self.scan_path + 'jones_guess_avg_r256_p256_f1024.npy' )

    def load_data(self):
        
        print("************* setting-up data structure")
        
        # the data array is the full set of measurements
        
        self.data = np.zeros((self.npts, self.pmodes, self.amodes, self.padpix, self.padpix))
        self.data_energy = np.zeros((self.npts, self.pmodes, self.amodes))
    
        print("************* loading data")
       
        # ... load data into data array ... #
        # ... don't need to calculate the jones matrices for the polarisers ... #
        # ... they will be included in the amp and phase of the probes ... #
        
        for i in range(self.pmodes):
            for j in range(self.amodes):
                
                print("polarisation mode: {}|analyser mode: {}".format(i,j))
                
                kk = (i*self.pmodes) + j
                
                self.subdir = "scan_%02d/grid_00_00/" % np.int(self.vpie_config[kk,0])
                
                self.theta_p[i] = self.vpie_config[kk,1]
                self.theta_a[j] = self.vpie_config[kk,2]		
                
                for k in np.arange(self.npts):
                    
                    self.procdir = "processed_r" + str(self.rebpix) + "_p" + str(self.padpix) + "/npy_files/"
                    
                    filename = "position_%04d_int.npy" % k
                    
                    self.full_filename = self.scan_path + self.subdir + self.procdir + filename
        
                    self.data[k,i,j,self.spx:self.epx,self.spx:self.epx] = np.sqrt(np.load(self.full_filename))
                    
                    self.data_energy[k,i,j] = np.sum(self.data[k,i,j,self.spx:self.epx,self.spx:self.epx]**2 )
        
                    self.data[k,i,j] = np.fft.fftshift(self.data[k,i,j])
        
        # ... transform angles from degrees to radians
        self.theta_p = np.radians(self.theta_p)
        self.theta_a = np.radians(self.theta_a)

        self.stheta_p = np.sin( self.theta_p )
        self.ctheta_p = np.cos( self.theta_p )
        
        self.stheta_a = np.sin( self.theta_a )
        self.ctheta_a = np.cos( self.theta_a )
        
        #for a given position on the sample, the detector estimate will depend the polariser and analyser settings

        #thus we need to declare psi_det_est to have pmodes*amodes slices
        
        # ... load the sample positions text file
        spec_data = sp.loadtxt(self.scan_path + ("scan_%02d" % np.int(self.vpie_config[0,0])) + '/grid_00_00/final_smarpod_positions.txt', delimiter=',' )
    
        
        # ... and create a new set of position arrays in pixel units
        pos_x = spec_data[:,0] / self.dx4
        pos_y = spec_data[:,1] / self.dx4
        
        self.ix = np.round( pos_x ) 
        self.iy = np.round( pos_y )
        
        # this is important ... the probe is always in the middle of the smaller array ... the rx and ry tell the algorithm where to put the top corner of the cutout (sample) array so you're cutting out the right part
        
        self.rx = self.sammid - self.padmid
        self.ry = self.sammid - self.padmid
        
        self.fft_denom = 1./self.padpix

        
        










    def guess_probe(self):
        
        
        beam = np.zeros((2, self.padpix, self.padpix)).astype(self.c_dtype)
        
        beam[0] = np.load(self.probe_path + 'sim_probe_guess_r256_p256.npy')
        
        radpix = sp.hypot(*sp.ogrid[-self.padmid:self.padmid,-self.padmid:self.padmid])
        phase_factor = sp.exp((self.J*self.PI/self.WL) * ( self.z43**(-1.) * (self.dx4 * radpix)**2.) )
        prop_factor = sp.exp((self.J*self.PI/self.WL) * ( ( self.z23**-1 + self.z34**(-1.) ) * (self.dx4 * radpix)**2.) ).astype(self.c_dtype)
        
        self.probe = np.zeros((self.pmodes, 2, self.padpix, self.padpix)).astype(self.c_dtype)
        self.probe_conj = np.zeros((self.pmodes, 2, self.padpix, self.padpix)).astype(self.c_dtype)
        
        
        '''
        beam = np.zeros((padpix,padpix))
        beamwidth = 10e-6
        mult = -1.0 / (2.0*beamwidth**2)
        beam = np.zeros((2, padpix, padpix)).astype(c_dtype)
        beam[0] = np.exp( mult * (radpix*d1)**2 )
        probe_guess = np.zeros((pmodes, 2, padpix, padpix)).astype(c_dtype)
        probe_conj = np.zeros((pmodes, 2, padpix, padpix)).astype(c_dtype)
        '''
        
        for i in range(self.pmodes):
            self.probe[i] = jones.prop_through_hwp(beam, self.theta_p[i]/2.)
            
            self.probe_conj[i,0] = np.conj(self.probe[i,0] )
            self.probe_conj[i,1] = np.conj(self.probe[i,1] )
        
    def analvect(self, amode):
        avect = np.array([np.cos(amode), np.sin(amode)])
        return avect    
    def iterate(self):
        
        self.ddbeta = 0.25
        self.ddbetavec = np.zeros( self.iterations )
        
        self.ddbetavec[0*self.iterations//4 : 1*self.iterations//4] = 0.5
        self.ddbetavec[1*self.iterations//4 : 2*self.iterations//4] = 0.6
        self.ddbetavec[2*self.iterations//4 : 3*self.iterations//4] = 0.7
        self.ddbetavec[3*self.iterations//4 : 4*self.iterations//4] = 0.8
        
        # probe[mode, plane] so probe[0,1] = probe[0th mode, y-plane] and probe[2,0] = probe[2nd mode, x-plane]
        self.rho_xx_max = np.max(np.abs(self.probe[0,0])**2 + np.abs(self.probe[1,0])**2 + np.abs(self.probe[2,0])**2)
        self.rho_yy_max = np.max(np.abs(self.probe[0,1])**2 + np.abs(self.probe[1,1])**2 + np.abs(self.probe[2,1])**2)
        

        
        for ii in np.arange(self.nruns):

            trans_tmp = np.zeros((2, 2, self.padpix, self.padpix)).astype(self.c_dtype)
            
            #pdb.set_trace()
            
            # loop over the number of iterations
            for i in np.arange( self.iterations ):
                
                rand_pos = np.random.permutation( self.ptych_num )
                
                #pdb.set_trace()
                
                # loop over the number of scan points
                for j in np.arange( self.ptych_num ):
                    
                    jj = rand_pos[ j ]
            
                    # work out the ROI on the larger, object array that corresponds to the k^th projection
                    x_region = np.int( self.rx + self.ix[ jj ] )
                    y_region = np.int( self.ry + self.iy[ jj ] )
            
                    xi = x_region
                    xf = x_region + self.padpix
                    yi = y_region
                    yf = y_region + self.padpix
                    
                    # crop out the relevant part of the sample
                    trans_crop = np.copy(self.trans[ : , : , yi : yf , xi : xf ])
                    
                    # loop over the number of incident polarisation settings
                    for k in np.arange( self.pmodes ):
                        
                        self.esw = jones.jones_product(trans_crop,self.probe[k])
                        
                        # loop over the number of analyser settings
                        for l in np.arange( self.amodes ):
                            
                            # store the diffraction data in a temp array
                            #temp_diff_amp = data[k,l,jj]
                            temp_diff_amp = self.data[jj,k,l]
        
                            # propagate through the analyser
                            self.aESW = jones.prop_through_lin_pol(self.esw, self.theta_a[l])
                            
                            # we know the field is linearly polarised
                            # so change coords from (x,y) to (u,v) such that the field is polarised along u
                            # this allows us to represent the field in a scalar formalism ( only amp/phase )
                            #scaESW = jones.rotate_coords(self.aESW, self.theta_a[l])[0]
                            scaESW = jones.rotate_coords(self.aESW, self.theta_a[l])[0]
                            # propagate to the detector
                            ff_meas = optics.downstream_prop(scaESW)
                            
                            # copy into a temporary array
                            #ff_meas = np.copy(psi_det_est[k,l])
                            
                            threshval = 0.001 * np.max(np.abs(ff_meas))
                            
                            # apply the modulus constraint
                            ft_guess = sp.where( ne.evaluate("real(abs(ff_meas))") > threshval, ne.evaluate("real(abs(temp_diff_amp))*(ff_meas/real(abs(ff_meas)))") , 0.0 ).astype(self.c_dtype )
                            
                            # calculate the complex difference
                            self.cplx_diff[k,l] = ft_guess - ff_meas
                            
                            # propagate the difference back to the exit surface of the analyser
                            #psi_analyser_est[k,l] = spf.fftshift( spf.ifft2( spf.fftshift( cplx_diff ) ) ) * padpix

                
                
                            
                            
                            
                temp_arr1 = (self.ctheta_a[0]*self.cplx_diff[k,0]) + (self.ctheta_a[1]*self.cplx_diff[k,1]) + (self.ctheta_a[2]*self.cplx_diff[k,2])
                self.arr_A[k,:,:] = optics.upstream_prop(temp_arr1)
                
                temp_arr2 = (self.stheta_a[0]*self.cplx_diff[k,0]) + (self.stheta_a[1]*self.cplx_diff[k,1]) + (self.stheta_a[2]*self.cplx_diff[k,2])
                self.arr_B[k,:,:] = optics.upstream_prop(temp_arr2)
            

            
                trans_tmp[0,0] = trans_crop[0,0] + (self.ddbetavec[i]/self.rho_xx_max) * ( (self.probe_conj[0,0]*self.arr_A[0]) + (self.probe_conj[1,0]*self.arr_A[1]) + (self.probe_conj[2,0]*self.arr_A[2]) )
                trans_tmp[0,1] = trans_crop[0,1] + (self.ddbetavec[i]/self.rho_yy_max) * ( (self.probe_conj[0,1]*self.arr_A[0]) + (self.probe_conj[1,1]*self.arr_A[1]) + (self.probe_conj[2,1]*self.arr_A[2]) )
                trans_tmp[1,0] = trans_crop[1,0] + (self.ddbetavec[i]/self.rho_xx_max) * ( (self.probe_conj[0,0]*self.arr_B[0]) + (self.probe_conj[1,0]*self.arr_B[1]) + (self.probe_conj[2,0]*self.arr_B[2]) )
                trans_tmp[1,1] = trans_crop[1,1] + (self.ddbetavec[i]/self.rho_yy_max) * ( (self.probe_conj[0,1]*self.arr_B[0]) + (self.probe_conj[1,1]*self.arr_B[1]) + (self.probe_conj[2,1]*self.arr_B[2]) )
                self.trans[ : , : , yi : yf , xi : xf ] = trans_tmp
                
                for j in range(self.ptych_num):
                    
                    jj = rand_pos[ j ]
            
                    # work out the ROI on the larger, object array that corresponds to the k^th projection
                    x_region = np.int( self.rx + self.ix[ jj ] )
                    y_region = np.int( self.ry + self.iy[ jj ] )
            
                    xi = x_region
                    xf = x_region + self.padpix
                    yi = y_region
                    yf = y_region + self.padpix
                    
                    for k in range(self.pmodes):
                        
                        
                        for l in range(self.amodes):
                            print(self.trans.shape)
                            obj_k = self.trans[ : , : , yi : yf , xi : xf]
                            print(obj_k.shape)
                            temp_probe = np.zeros(obj_k.shape)
                            
                            print(ff_meas.shape)
                            print(self.analvect(l).shape)
                            print(obj_k.shape)
                            
                            
                            delta_p = np.conj(obj_k).T*optics.upstream_prop(np.cos(l)*ff_meas+np.sin(l)*ff_meas)
                            print(delta_p.shape)
                        
                        
                
                
                """ update probe """
                #h = np.array([np.sum(abs(self.ctheta_a)**2),
                #              np.sum(self.ctheta_a*np.conj(self.stheta_a)),
                #              np.conj(np.sum(self.ctheta_a*np.conj(self.stheta_a))),
                #              np.sum(abs(self.stheta_a)**2)])
                
                #h = h.reshape([2,2])
                
                
                #Dn = np.diag(np.sum(np.conj(np.sum(self.trans.T)))*h*np.sum(self.trans))
                
                #self.probe = self.probe - self.ddbetavec[i]/Dn
                
                #print("iteration: {}".format(i))
                #plt.imshow(np.imag(self.trans[0,0]))
                plt.imshow(np.imag(self.probe))
                plt.show()
            current_fname = "jones_guess_%02d" % ii + "_r" + str(self.rebpix) + "_p" + str(self.padpix) + "_f" + str(self.sampix) + ".npy"

            
        
            """
            include probe update  somewhere around here.
            """
        
def run():
    PIE = vPIE('/opt/data/sim_data/henry_02/', iterations = 2)
    PIE.load_data()
    PIE.guess_probe()
    PIE.iterate()
    print("test complete")
run()
       





#sample_est = np.load('/home/guido/data/objects/birefringent_wheel/jones_matrix_2048_x_2048.npy')



            


