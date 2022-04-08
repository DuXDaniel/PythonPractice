import numpy as np
import scipy as sp
import scipy.integrate as spi
import scipy.interpolate as spinterp
import math as math
import statsmodels as sm
import sklearn as skl
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import tqdm
import multiprocessing as mlpr
import sys
from itertools import repeat

def temporal_gauss(z,t,sig_las):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    val = np.exp(-(2*(z-c*t)**2)/(sig_las**2*c**2))
    return val

def omeg_las_sq(z, beam_waist):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    w0 = beam_waist # nm
    z0 = np.pi*w0**2/lam # Rayleigh range, nm
    val = w0**2*(1+z**2/z0**2)
    return val

def spatial_gauss(rho_xy,z,t, beam_waist,sig_las):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    w0 = beam_waist # nm
    z0 = np.pi*w0**2/lam # Rayleigh range, nm
    val = 1/np.pi/omeg_las_sq(z, beam_waist)*np.exp(-(2*rho_xy**2)/(omeg_las_sq(z, beam_waist)/temporal_gauss(z,t,sig_las)))
    return val

def laser(rho_xy,z,t, beam_waist,sig_las):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    w0 = beam_waist # nm
    z0 = np.pi*w0**2/lam # Rayleigh range, nm
    val = spatial_gauss(rho_xy,z,t, beam_waist,sig_las)*temporal_gauss(z,t,sig_las)
    return val

def norm_laser_integrand(rho_xy,z,t,beam_waist,sig_las):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    w0 = beam_waist # nm
    z0 = np.pi*w0**2/lam # Rayleigh range, nm
    val = 2*np.pi*rho_xy*laser(rho_xy,z,t, beam_waist,sig_las)
    return val

def laser_sum(t, gauss_limit, sig_las, beam_waist):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    w0 = beam_waist # nm
    z0 = np.pi*w0**2/lam # Rayleigh range, nm
    val = spi.dblquad(norm_laser_integrand, -gauss_limit*sig_las + c*t, gauss_limit*sig_las + c*t, 0, gauss_limit*np.sqrt(omeg_las_sq(c*t, beam_waist)), args=[t, beam_waist,sig_las])
    return val

def main(argv):
    voxel_granularity = 81
    slice_granularity = 81
    gauss_limit = 3

    #input('begin variable defs, press to continue')

    print('Seeding workspace with relevant information.')

    #input('begin variable defs, press to continue')

    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    w0 = 100e3 # nm
    sig_las = 1e3 # ps
    z0 = np.pi*w0**2/lam # Rayleigh range, nm

    #input('begin function defs, press to continue')

    E_photon = planck*c/lam # nJ
    E_pulse = 1 # nJ
    n_pulse = E_pulse/E_photon # number of photons per pulse

    xover_slope = 3e-3/300e-3 # 3 mm in 300 mm of travel
    xover_angle = np.arctan(xover_slope) # degrees
    vel = 2.33e8*1e9/1e12 # velocity of electron, 200 kV, nm./ps
    sig_ebeam = 1e3 # time resolution of ebeam, ps

    def omeg_ebeam(y):
      val = np.absolute(xover_slope*y)
      return val

    def e_beam_xz(rho_xz,y):
      e_beam_xz = 1/np.pi/(omeg_ebeam(y))**2*np.exp(-(2*rho_xz**2)/(omeg_ebeam(y))**2)
      return val

    def e_beam_xz_raster(x,y,z):
      val = 1/np.pi/(omeg_ebeam(y))**2*np.exp(-(2*(x**2 + z**2))/(omeg_ebeam(y))**2)
      return val

    def e_beam_yt(y):
      val = 1/np.pi/sig_ebeam**2/vel**2*np.exp(-(2*(y)**2)/(sig_ebeam**2*vel**2))
      return val

    #input('functions defined, press to continue')

    # --> Key: don't need to calculate instantaneous density of electrons, just
    # need the path and the ending normalization.

    ## establishing interpolated normalization for laser
    sig_arr = np.array([sig_ebeam,sig_las])
    t_spacing = sig_arr.min()/slice_granularity
    t_limit = gauss_limit*sig_ebeam
    t_range = np.arange(-t_limit,t_limit+t_spacing,t_spacing)
    laser_sum_array = np.zeros_like(t_range)
    laser_sum_err = np.zeros_like(t_range)

    #input('beginning integral')
    for i in np.arange(t_range.size):
      #print(i)
      #print(t_range[i])
      #print(laser_sum_array[i])
      #input('progressing integral')
      #print(laser_sum(t_range[i]))
      val_int = laser_sum(t_range[i], gauss_limit, sig_las, w0)
      laser_sum_array[i] = val_int[0]
      laser_sum_err[i] = val_int[1]

    select_zero_laser_sum_array = np.where(laser_sum_array == 0, 1, 0)
    norm_factor_array = 1/(select_zero_laser_sum_array*1e308 + laser_sum_array)

    #### Construct voxels
    # how granular do I want the voxels if they all pass through (0,0) at
    # crossover? It's arbitrary at that point. Let us assume a 1000 x 1000
    # grid split across 6-sigma centered at the peak

    #e_beam_xz_norm = 1/np.quad(integral(e_beam_xz(x,3*sig_ebeam*vel), 0, np.inf))

    #### Construct slices
    # Do initial normalization, then segment into the slices slices dependent
    # on e_pulse res (let's assume 1000 slices?) or perhaps make it such that
    # it is the smaller of the laser or e pulse resolutions divided by 10000 as
    # the spacing?

    e_beam_int = spi.quad(e_beam_yt, -np.inf, np.inf)
    e_beam_yt_norm = 1/e_beam_int[0]

    # --> Normalize on a per slice basis here
    # --> which is to say, ignore the spatial component and normalize the
    # temporal./y component and attach each slice a portion of that
    # normalization
    # --> Then simply multiply that slice's normalization factor to the
    # individual voxel of each slice at the maximum expansion point after
    # Compton scattering. Just take the distribution at t_end.

    ## calculating voxel weights... need to find integration bounds for each voxel point at the final position of the e-beam ("detector")
    # voxel_xz_grid_weights = zeros(voxel_granularity, voxel_granularity);
    voxel_y_weights = np.zeros(slice_granularity)

    sig_ratios = np.array([-gauss_limit, -1.6449, -1.2816, -1.0364, -0.8416, -0.6745, -0.5244, -0.3853, -0.2533, -0.1257, -0.0627]) # limit, 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05
    weights = np.array([0.045, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025])

    if (sig_ebeam < sig_las):
      t_bounds_overall = []
      sig_ratios = np.insert(sig_ratios,0,-gauss_limit*sig_las/sig_ebeam)
      weights = np.insert(weights,0,0.005)
      num_voxels = 0
      for i in np.arange(len(t_bounds_overall)-1):
          t_bounds_add = np.linspace(sig_ratios[i]*sig_ebeam,sig_ratios[i+1]*sig_ebeam,round(weights[i]*slice_granularity))
          t_bounds_overall.append(t_bounds_add[1:-2])
          num_voxels = num_voxels + len(t_bounds_overall[i])

      voxels_left = slice_granularity + 1 - num_voxels*2
      t_bounds_overall.append(np.linspace(sig_ratios[-1]*sig_ebeam,-sig_ratios[-1]*sig_ebeam,voxels_left))
    elif (sig_las < sig_ebeam):
      t_bounds_overall = []
      sig_ratios = np.insert(sig_ratios,0,-gauss_limit*sig_las/sig_ebeam)
      weights = np.insert(weights,0,0.005)
      num_voxels = 0
      for i in np.arange(len(t_bounds_overall)-1):
          t_bounds_add = np.linspace(sig_ratios[i]*sig_las,sig_ratios[i+1]*sig_las,round(weights[i]*slice_granularity))
          t_bounds_overall.append(t_bounds_add[1:-2])
          num_voxels = num_voxels + len(t_bounds_overall[i])

      voxels_left = slice_granularity + 1 - num_voxels*2;
      t_bounds_overall.append(np.linspace(sig_ratios[-1]*sig_las,-sig_ratios[-1]*sig_las,voxels_left))
    else:
      t_bounds_overall = [];
      num_voxels = 0;
      for i in np.arange(len(t_bounds_overall)-1):
          t_bounds_add = np.linspace(sig_ratios[i]*sig_ebeam,sig_ratios[i+1]*sig_ebeam,round(weights[i]*slice_granularity))
          t_bounds_overall.append(t_bounds_add[1:-2])
          num_voxels = num_voxels + len(t_bounds_overall[i])

      voxels_left = slice_granularity + 1 - num_voxels*2
      t_bounds_overall.append(np.linspace(sig_ratios[-1]*sig_ebeam,-sig_ratios[-1]*sig_ebeam,voxels_left))

    t_bounds = np.array([]);
    for i in np.arange(len(t_bounds_overall)):
      t_bounds = np.append(t_bounds,t_bounds_overall[i])

    for i in np.flip(np.arange(len(t_bounds_overall)-1)):
      t_bounds = np.append(t_bounds,np.flip(-t_bounds_overall[i]))

    y_bounds = t_bounds*vel

    #y_bounds = linspace(-3.*sig_ebeam.*vel - y_spacing./2,3.*sig_ebeam.*vel + y_spacing./2,voxel_granularity+1);

    for l in np.arange(slice_granularity):
      weight_int = spi.quad(e_beam_yt, y_bounds[l], y_bounds[l+1])
      voxel_y_weights[l] = e_beam_yt_norm*weight_int[0]

    # voxel_y_weights = xlsread('Y_Weights.xlsx');

    #### Construct pathway of travel for all voxels
    # pathway of travel is identical at all positions of xz, so just find a
    # generic pathway of travel for a double cone base the pathway off of the
    # crossover angle (just define by slope maybe?) --> Seems like another
    # variable to take note of

    # Need to calculate changing width of e-beam

    #### voxel square sizes can be same because of travel path. Just change
    # magnitude of Gaussian from back focal plane

    # How to populate location data? Want the lower half of the e-beam
    # (symmetry) and 3-sigma from the center
    # can populate at time zero and then let integral back-calculate from -Inf.
    voxel_grid_phase_data = np.zeros((voxel_granularity,voxel_granularity,slice_granularity))
    voxel_grid_slope_x_data = np.zeros((voxel_granularity,voxel_granularity)) # travel path information for each voxel, only need one representative plane for each
    voxel_grid_slope_z_data = np.zeros((voxel_granularity,voxel_granularity)) # travel path information for each voxel, only need one representative plane for each
    voxel_grid_y_data = np.zeros((voxel_granularity,voxel_granularity,slice_granularity))

    y_range = np.zeros(slice_granularity)

    for l in np.arange(slice_granularity):
      voxel_grid_y_data[:,:,l] = np.ones((voxel_granularity,voxel_granularity))*((y_bounds[l]+y_bounds[l+1])/2)
      y_range[l] = (y_bounds[l]+y_bounds[l+1])/2

    if (sig_ebeam < sig_las):
      y_dist_from_center = gauss_limit*sig_las*vel
      for j in np.arange(voxel_granularity):
          voxel_grid_slope_x_data[j,:] = np.linspace(-gauss_limit*omeg_ebeam(gauss_limit*sig_las*vel),gauss_limit*omeg_ebeam(gauss_limit*sig_las*vel),voxel_granularity)/y_dist_from_center
          voxel_grid_slope_z_data[:,j] = np.linspace(-gauss_limit*omeg_ebeam(gauss_limit*sig_las*vel),gauss_limit*omeg_ebeam(gauss_limit*sig_las*vel),voxel_granularity)/y_dist_from_center
    elif (sig_las <= sig_ebeam):
      y_dist_from_center = gauss_limit*sig_ebeam*vel
      for j in np.arange(voxel_granularity):
          voxel_grid_slope_x_data[j,:] = np.linspace(-gauss_limit*omeg_ebeam(gauss_limit*sig_ebeam*vel),gauss_limit*omeg_ebeam(gauss_limit*sig_ebeam*vel),voxel_granularity)/y_dist_from_center
          voxel_grid_slope_z_data[:,j] = np.linspace(-gauss_limit*omeg_ebeam(gauss_limit*sig_ebeam*vel),gauss_limit*omeg_ebeam(gauss_limit*sig_ebeam*vel),voxel_granularity)/y_dist_from_center


    ## finding integral bounds

    integral_bound = min([gauss_limit*z0/c, t_limit])

    ## Loop

    num_voxels = slice_granularity*voxel_granularity**2
    voxel_grid_phase_data_unpacked = np.zeros(num_voxels)

    init_y_vals = np.zeros(num_voxels)
    x_slopes = np.zeros(num_voxels)
    z_slopes = np.zeros(num_voxels)
    for cur_voxel in np.arange(num_voxels):
        cur_slice = math.floor(i/voxel_granularity**2)
        cur_voxel_num = i % (voxel_granularity**2)
        m = math.floor(cur_voxel_num/voxel_granularity)
        n = cur_voxel_num % voxel_granularity

        init_y_vals[cur_voxel] = voxel_grid_y_data[m,n,cur_slice]
        x_slopes[cur_voxel] = voxel_grid_slope_x_data[m,n]
        z_slopes[cur_voxel] = voxel_grid_slope_z_data[m,n]

    #pool_obj = mlpr.Pool(processes = mlpr.cpu_count()-1)

    pbar = tqdm.tqdm(total=num_voxels)

    t = time.time()
    #with pool_obj:
    #    voxel_grid_phase_data_unpacked = pool_obj.starmap(calc_func,zip(np.arange(num_voxels),init_y_vals,x_slopes,z_slopes,repeat(voxel_granularity)))

    for i in np.arange(num_voxels):
        cur_slice = math.floor(i/voxel_granularity**2)
        cur_voxel_num = i % (voxel_granularity**2)
        m = math.floor(cur_voxel_num/voxel_granularity)
        n = cur_voxel_num % voxel_granularity
        calc = calc_func(i,init_y_vals[i],x_slopes[i],z_slopes[i],voxel_granularity,vel,w0,t_range,norm_factor_array,integral_bound)
        voxel_grid_phase_data[m,n,cur_slice] = calc
        pbar.update(1)

    elapsed = time.time() - t
    print(elapsed)
    #pool_obj.close()
    #pool_obj.join()
    pbar.close()
    # generate map distribution of electron beam, summing and averaging over
    # all slices
    '''
    for cur_voxel in np.arange(num_voxels):
        cur_slice = math.floor(i/voxel_granularity**2)
        cur_voxel_num = i % (voxel_granularity**2)
        m = math.floor(cur_voxel_num/voxel_granularity)
        n = cur_voxel_num % voxel_granularity
        voxel_grid_phase_data[m,n,cur_slice] = voxel_grid_phase_data_unpacked[cur_voxel]
    '''
    final_phase_data = np.zeros((voxel_granularity, voxel_granularity))
    #stdev_phase_data = zeros(voxel_granularity, voxel_granularity);

    # assignin('base','final_phase_data',final_phase_data);
    # assignin('base','voxel_grid_phase_data',voxel_grid_phase_data);
    # assignin('base','voxel_y_weights',voxel_y_weights);

    for m in np.arange(voxel_granularity):
      for n in np.arange(voxel_granularity):
          for p in np.arange(slice_granularity):
              final_phase_data[m,n] = final_phase_data[m,n] + voxel_grid_phase_data[m,n,p]*voxel_y_weights[p]

    input('awaiting')

def calc_func(cur_point,init_y_val,x_slope,z_slope,voxel_granularity,vel, beam_waist,t_range,norm_factor_array,integral_bound):
    voxel_granularity = 81
    slice_granularity = 81
    gauss_limit = 3

    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    sig_las = 1e3 # ps

    E_photon = planck*c/lam # nJ
    E_pulse = 1 # nJ
    n_pulse = E_pulse/E_photon # number of photons per pulse

    xover_slope = 3e-3/300e-3 # 3 mm in 300 mm of travel
    xover_angle = np.arctan(xover_slope) # degrees
    vel = 2.33e8*1e9/1e12 # velocity of electron, 200 kV, nm./ps
    sig_ebeam = 1e3 # time resolution of ebeam, ps
    t_limit = gauss_limit*sig_ebeam

    cur_slice = math.floor(cur_point/voxel_granularity**2)
    cur_voxel_num = cur_point % (voxel_granularity**2)
    m = math.floor(cur_voxel_num/voxel_granularity)
    n = cur_voxel_num % voxel_granularity

    # Determine slice level --> will determine weighting at the end

    # Assumption: all electrons pass through (x0,z0) at crossover most
    # likely incorrect, but we have nothing else to go off of will only
    # slightly cause the CTF to show a higher than normal resolution

    # reference current voxel xz position grid from travel path
    # calculation and current time

    # calculate photon densities at position grid

    # calculate path for current x(t), y(t), z(t) for specific slice, and
    # voxel m,n. This is the path of the electron, but these values
    # are placed into the laser equation.

    y_func = lambda t: init_y_val-vel*t
    x_func = lambda t: y_func(t)*x_slope
    z_func = lambda t: y_func(t)*z_slope
    rho_func = lambda t: np.sqrt(x_func(t)**2 + y_func(t)**2)

    ##
    interp_norm = spinterp.interp1d(t_range,norm_factor_array)
    norm_factor = lambda t: (np.absolute(t) <= t_limit)*interp_norm(t)
    full_func = lambda t: hbar*alpha*n_pulse*norm_factor(t)*laser(rho_func(t),z_func(t),t, beam_waist,sig_las)*lam/np.sqrt(mass_e**2*(1+vel**2/c**2))

    ##
    integral_waypoints = [-np.absolute(init_y_val)/vel,0,np.absolute(init_y_val)/vel]

    val = spi.quad(full_func, -integral_bound, integral_bound, points=integral_waypoints)
    return (not math.isnan(val[0]))*val[0]

if __name__ == '__main__':
    main(sys.argv)
