"""


iterative ptychographical reconstruction assuming vector wavefields


the algorithm is developed using the theory outlined in,

"Ptychography in anisotropic media, Ferrand, Allain, Chamard, 2015"


"""

import jones_matrices as jones

import numexpr as ne
import numpy as np


import optics_utils as optics


import pylab as pyl

import scipy as sp

from math import cos, sin

from tqdm import tqdm, trange
from matplotlib import pyplot as plt

from vis_utils import plot_probe, plot_obj

from copy import copy
from sklearn.preprocessing import minmax_scale as norm

# from opt.misc.misc import check_nan

pi = np.pi


class config:
    def __init__(
        self,
        scan_path,
        pmodes=3,
        amodes=3,
        iterations=1,
        nruns=1,
        npts=360,
        wav=660e-09,
        det_pix=2048,
        pixdx=6.5e-06,
        bit_precision=32,
        vpie_config_num=1,
    ):

        """
        :param scan_path: data path
        :param pmodes: number of polarisation modes of the probe
        :param amodes: number of analyser modes for detection
        :param iterations: number of iterations of the reconst. algorithm
        :param nruns: instances of the reconst. algorithm
        :param npoints: number of scan points
        :param wav: wavelength of incident light
        :param det_pix: number of detector pixel
        :param pixdx: detector pixel resolution
        :param bit_precision: data bit precision for dtypes
        :param vpie_config_num: data config number
        """

        self.bit_precision = bit_precision

        self.r_dtype = "float{}".format(bit_precision)
        self.c_dtype = "complex{}".format(bit_precision * 2)

        self.pmodes = pmodes
        self.amodes = amodes
        self.nruns = nruns
        self.iterations = iterations
        self.scan_path = scan_path
        self.npts = npts
        self.wav = wav

        self.vpie_config = np.loadtxt(
            scan_path + "vpie_config_%02d.txt" % vpie_config_num
        )

        self.det_pix = det_pix
        self.pixdx = pixdx

        self.rebfac = 8
        self.padfac = 1
        self.fovfac = 4

        self.rebpix = self.det_pix // self.rebfac
        self.rebmid = self.rebpix // 2
        self.rebqrt = self.rebmid // 2

        self.padpix = self.rebpix * self.padfac
        self.padmid = self.padpix // 2
        self.padqrt = self.padmid // 2

        self.sampix = self.padpix * self.fovfac
        self.sammid = self.sampix // 2
        self.samqrt = self.sammid // 2

        self.dx3 = self.rebfac * self.pixdx

        self.z12 = 0.25
        self.z23 = 0.30
        self.z24 = 0.04

        self.z43 = self.z23 - self.z24
        self.z34 = -self.z43

        self.dx2 = (self.wav * self.z23) / (self.padpix * self.dx3)
        self.dx1 = (self.wav * self.z12) / (self.padpix * self.dx2)
        self.dx4 = (self.wav * self.z43) / (self.padpix * self.dx3)

        self.spx = self.padmid - self.rebmid
        self.epx = self.padmid + self.rebmid


def load_data(config):

    """
    :param config: config object containing relevant experimental definitions
    
    :returns data: experimental data [scans, pmodes, amodes, x, y]
    """

    config.theta_p, config.theta_a = np.zeros(config.pmodes), np.zeros(config.amodes)
    data = np.zeros(
        (config.npts, config.pmodes, config.amodes, config.padpix, config.padpix)
    )
    data_energy = np.zeros((config.npts, config.pmodes, config.amodes))

    if type(config) == "str":
        print("input should be config file containing scan_path")

    elif config.scan_path is not None:
        print("loading from config")

        # the data array is the full set of measurements

        # ... load data into data array ... #
        # ... don't need to calculate the jones matrices for the polarisers ... #
        # ... they will be included in the amp and phase of the probes ... #

        for k in trange(config.pmodes, desc="loading data"):
            for l in range(config.amodes):

                config.subdir = "scan_%02d/grid_00_00/" % np.int(
                    config.vpie_config[k * config.pmodes + l, 0]
                )

                config.theta_p[k] = config.vpie_config[k * config.pmodes][1]
                config.theta_a[l] = config.vpie_config[k * config.pmodes + l][2]
                config.theta_p, config.theta_a = (
                    np.radians(config.theta_p),
                    np.radians(config.theta_a),
                )
                for j in np.arange(config.npts):

                    config.procdir = (
                        "processed_r"
                        + str(config.rebpix)
                        + "_p"
                        + str(config.padpix)
                        + "/npy_files/"
                    )

                    filename = "position_%04d_int.npy" % k

                    config.full_filename = (
                        config.scan_path + config.subdir + config.procdir + filename
                    )
                    data[
                        j, k, l, config.spx : config.epx, config.spx : config.epx
                    ] = np.sqrt(np.load(config.full_filename))

                    data_energy[j, k, l,] = np.sum(
                        data[j, k, l, config.spx : config.epx, config.spx : config.epx]
                        ** 2
                    )

                    data[j, k, l,] = np.fft.fftshift(data[j, k, l,])
 
        # for a given position on the sample, the detector estimate will depend the polariser and analyser settings

        # thus we need to declare psi_det_est to have pmodes*amodes slices

        # ... load the sample positions text file
        spec_data = sp.loadtxt(
            config.scan_path
            + ("scan_%02d" % np.int(config.vpie_config[0, 0]))
            + "/grid_00_00/final_smarpod_positions.txt",
            delimiter=",",
        )

        # ... and create a new set of position arrays in pixel units
        pos_x = spec_data[:, 0] / config.dx4
        pos_y = spec_data[:, 1] / config.dx4

        config.ix = np.round(pos_x)
        config.iy = np.round(pos_y)

        # this is important ... the probe is always in the middle of the smaller array ... the rx and ry tell the algorithm where to put the top corner of the cutout (sample) array so you're cutting out the right part

        config.rx = config.sammid - config.padmid
        config.ry = config.sammid - config.padmid

        config.fft_denom = 1.0 / config.padpix
        
        nans = np.isnan(data)
        data[nans] = 0
        return data


def calc_probe(config):

    """
    :param config: configuration object
    
    :returns probe: a priori probe estimate [pmodes, polarisation components, x, y]
    """

    cplx = complex(0, 1)  # complex component
    beam = np.zeros((2, config.padpix, config.padpix)).astype(config.c_dtype)

    beam[0] = np.load(config.scan_path + "sim_probe_guess_r256_p256.npy")

    radpix = sp.hypot(
        *sp.ogrid[-config.padmid : config.padmid, -config.padmid : config.padmid]
    )
    phase_factor = sp.exp(
        (cplx * np.pi / config.wav)
        * (config.z43 ** (-1.0) * (config.dx4 * radpix) ** 2.0)
    )
    prop_factor = sp.exp(
        (cplx * np.pi / config.wav)
        * ((config.z23 ** -1 + config.z34 ** (-1.0)) * (config.dx4 * radpix) ** 2.0)
    ).astype(config.c_dtype)

    probe = np.zeros((config.pmodes, 2, config.padpix, config.padpix)).astype(
        config.c_dtype
    )

    for i in range(config.pmodes):
        probe[i] = jones.prop_through_hwp(beam, config.theta_p[i] / 2.0)
    
    probe = np.random.rand(*probe.shape) + np.random.rand(*probe.shape)*1j
 
    #probe[:,:,100:200, 100:200] = 0
    
    return probe

def analysis_op(l):
    h = np.array([cos(l), sin(l)]).reshape([2,1])
    return h



def reconstruction(config, data, probe, update="object"):

    """
    main reconstruction algorithm for joint estimation of probe and object
    
    :param config: configuration object
    :param data: experimental data [scans, pmodes, amodes, x, y]
    :param probe: probe estimate [pmodes, polarisation components, x, y]
    
    :returns obj: object estimate 
    :returns probe: probe estimate
    """

    cplx_diff = np.zeros(
        (config.pmodes, config.amodes, config.padpix, config.padpix)
    ).astype(config.c_dtype)

    obj = np.load(config.scan_path + "jones_guess_avg_r256_p256_f1024.npy")
    #obj += np.random.rand(2,2,1024,1024)+np.random.rand(2,2,1024,1024)*1j
    
    config.ddbeta = 0.25
    config.ddbetavec = np.zeros(config.iterations)

    config.ddbetavec[0 * config.iterations // 4 : 1 * config.iterations // 4] = 0.25
    config.ddbetavec[1 * config.iterations // 4 : 2 * config.iterations // 4] = 0.01
    config.ddbetavec[2 * config.iterations // 4 : 3 * config.iterations // 4] = 0.01
    config.ddbetavec[3 * config.iterations // 4 : 4 * config.iterations // 4] = 0.01

    # probe[mode, plane] so probe[0,1] = probe[0th mode, y-plane] and probe[2,0] = probe[2nd mode, x-plane]
    config.rho_xx_max = np.max(
        np.abs(probe[0, 0]) ** 2 + np.abs(probe[1, 0]) ** 2 + np.abs(probe[2, 0]) ** 2
    )
    config.rho_yy_max = np.max(
        np.abs(probe[0, 1]) ** 2 + np.abs(probe[1, 1]) ** 2 + np.abs(probe[2, 1]) ** 2
    )
    
    delta = [np.zeros_like(probe)]
    grad = [0]
    #### START HERE
    for ii in np.arange(config.nruns):

        obj_tmp = np.zeros((2, 2, config.padpix, config.padpix)).astype(config.c_dtype)

        arr_A = np.zeros((config.pmodes, config.padpix, config.padpix)).astype(
            config.c_dtype
        )
        arr_B = np.zeros((config.pmodes, config.padpix, config.padpix)).astype(
            config.c_dtype
        )
        
        rand_pos = np.random.permutation(config.npts)

        # pdb.set_trace()
        delta_p = np.zeros(probe.shape).astype(config.c_dtype)
        
        # loop over the number of iterations
        for itr in range(config.iterations):
            

            D = 0
            # loop over the number of scan points
            for j in tqdm(np.arange(config.npts), desc="npts"):

                jj = rand_pos[j]

                # work out the ROI on the larger object array that corresponds to the k^th projection
                x_region = np.int(config.rx + config.ix[jj])
                y_region = np.int(config.ry + config.iy[jj])

                xi = x_region
                xf = x_region + config.padpix
                yi = y_region
                yf = y_region + config.padpix

                # crop out the relevant part of the sample
                trans_crop = obj[:, :, yi:yf, xi:xf]
                
                lx,ly = list(config.theta_a), list(config.theta_a)
                ly.reverse()
                
                axx, ayx, axy, ayy = 0,0,0,0
                for l in range(len(lx)):
                    axx += np.abs(analysis_op(lx[l])[0]**2)
                    ayy += np.abs(analysis_op(lx[l])[1]**2)
                    axy += analysis_op(lx[l])[1]*np.conj(analysis_op(lx[l])[0])
                    ayx += analysis_op(lx[l])[0]*np.conj(analysis_op(lx[l])[1])
                

                f = np.array([axx, ayx, axy, ayy]).reshape([2,2])
                
                D_ = np.conj(trans_crop).T*f #scaling factor
                D_[:,:,0,0] = D_[:,:,0,0]*trans_crop[0,0]
                D_[:,:,1,1] = D_[:,:,0,0]*trans_crop[1,1]
                
                D += np.diag(np.array([D_[:,:,0,0], 0, 0, D_[:,:,1,1]]).reshape(2,2))

                # loop over the number of incident polarisation settings
                for k in np.arange(config.pmodes):
          
                    esw = jones.jones_product(trans_crop, probe[k])
 
                    # loop over the number of analyser settings
                    for l in np.arange(config.amodes):
                        
                        # store the diffraction data in a temp array
                        # propagate through the analyser
                        aESW = jones.prop_through_lin_pol(esw, config.theta_a[l])

                        # we know the field is linearly polarised
                        # so change coords from (x,y) to (u,v) such that the field is polarised along u
                        # this allows us to represent the field in a scalar formalism ( only amp/phase )
                        scaESW = jones.rotate_coords(aESW, config.theta_a[l])[0]
                        # propagate to the detector
                        ff_calc = optics.downstream_prop(scaESW)

                        ff_meas = copy(data[
                            jj, k, l,
                        ])

                             
                        # MODULUS CONSTRAINT                         
                        ff_meas = ff_meas.astype(config.c_dtype)
                        ff_meas.imag = ff_calc.imag
                        
 
                        
                        
                        ff_diff = ff_meas-ff_calc

                        int_meas = np.abs(ff_meas)**2

                        int_calc = np.abs(ff_calc)**2
                        
                        mod = np.sqrt(int_meas/int_calc)-1
                     
                        if update == "object" or "both":
                            # calculate the complex difference
                            #cplx_diff[k, l] = ff_calc[0] - ff_meas[0]
                            pass
                        
                        if update == "probe" or "both":
                            delta_p[k, 0] += -2*np.conj(
                                trans_crop[0, 0]
                            ).T * optics.upstream_prop(mod*(
                                np.cos(l) * ff_diff + np.sin(l) * ff_diff
                            ))
                            delta_p[k, 0] += -2*np.conj(
                                trans_crop[1, 0]
                            ).T * optics.upstream_prop(mod*(
                                np.cos(l) * ff_diff + np.sin(l) * ff_diff
                            ))

                            delta_p[k, 1] += -2*np.conj(
                                trans_crop[1, 0]
                            ).T * optics.upstream_prop(mod*(
                                np.cos(l) * ff_diff + np.sin(l) * ff_diff
                            ))
                            delta_p[k, 1] += -2*np.conj(
                                trans_crop[1, 1]
                            ).T * optics.upstream_prop(mod*(
                                np.cos(l) * ff_diff + np.sin(l) * ff_diff
                            ))


            delta.append(delta_p)
            grad.append(abs(np.sum(delta_p)))
            print(grad)
            print(delta_p)
 
            probe = probe -  (-delta_p[:,:]+delta[itr][:,:])
            
            probe[:,0,] /= D[0]
            probe[:,1,] /= D[1]
            plot_probe(config, probe)
            
                 
# =============================================================================
#             if update == "object" or "both":
#                 temp_arr1 = (
#                     (cos(config.theta_a[0]) * cplx_diff[k, 0])
#                     + (cos(config.theta_a[1]) * cplx_diff[k, 1])
#                     + (cos(config.theta_a[2]) * cplx_diff[k, 2])
#                 )
#                 arr_A[k, :, :] = optics.upstream_prop(temp_arr1)
# 
#                 temp_arr2 = (
#                     (sin(config.theta_a[0]) * cplx_diff[k, 0])
#                     + (sin(config.theta_a[1]) * cplx_diff[k, 1])
#                     + (sin(config.theta_a[2]) * cplx_diff[k, 2])
#                 )
#                 arr_B[k, :, :] = optics.upstream_prop(temp_arr2)
# 
#                 obj_tmp[0, 0] = trans_crop[0, 0] + (
#                     config.ddbetavec[itr] / config.rho_xx_max
#                 ) * (
#                     (np.conj(probe)[0, 0] * arr_A[0])
#                     + (np.conj(probe)[1, 0] * arr_A[1])
#                     + (np.conj(probe)[2, 0] * arr_A[2])
#                 )
#                 obj_tmp[0, 1] = trans_crop[0, 1] + (
#                     config.ddbetavec[itr] / config.rho_xx_max
#                 ) * (
#                     (np.conj(probe)[0, 1] * arr_A[0])
#                     + (np.conj(probe)[1, 1] * arr_A[1])
#                     + (np.conj(probe)[2, 1] * arr_A[2])
#                 )
#                 obj_tmp[1, 0] = trans_crop[1, 0] + (
#                     config.ddbetavec[itr] / config.rho_xx_max
#                 ) * (
#                     (np.conj(probe)[0, 0] * arr_A[0])
#                     + (np.conj(probe)[1, 0] * arr_A[1])
#                     + (np.conj(probe)[2, 0] * arr_A[2])
#                 )
#                 obj_tmp[1, 1] = trans_crop[1, 1] + (
#                     config.ddbetavec[itr] / config.rho_xx_max
#                 ) * (
#                     (np.conj(probe)[0, 1] * arr_A[0])
#                     + (np.conj(probe)[1, 1] * arr_A[1])
#                     + (np.conj(probe)[2, 1] * arr_A[2])
#                 )
# 
#                 obj[:, :, yi:yf, xi:xf] = obj_tmp
#                 
#     
#                 #if update == "probe" or "both":
#                 
#                 ### to be extended in the immediate future
#                 plot_obj(config, obj)
# =============================================================================
    return obj, probe


if __name__ == "__main__":

    print("Testing internal operations")

    config = config("/opt/data/sim_data/henry_02/", iterations=100,npts = 15)
    print("config class: Pass")

    data = load_data(config)
    print("data loading: Pass")

    probe = calc_probe(config)

    print("initialising probe guess: Pass")

    obj, probe = reconstruction(config, data, probe, update="both")
    print(obj.shape)
    plot_probe(config, probe)
    plot_obj(config, obj)
