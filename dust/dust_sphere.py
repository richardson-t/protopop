import numpy as np
import h5py
from astropy import constants,units as u
from astropy.modeling.models import BlackBody
from os import path

datapath = path.dirname(__file__)

def P_internal(M_cl,efficiency,T,R_cl,mu):
    M_res = M_cl * 100 / efficiency
    N = M_res / mu
    P = (3 * constants.k_B * N * T / 4 / np.pi / R_cl**3).to(u.J/u.cm**3)
    return P

def c_s(T,mu):
    return np.sqrt(constants.k_B * T / mu).to(u.km / u.s)

def BE_props(M_cl,efficiency,T,R_cl,mu):
    spd = c_s(T,mu)
    M_BE = 1.18 * spd**4 / np.sqrt(constants.G**3 *
                                   P_internal(M_cl,efficiency,T,R_cl,mu))
    rho_c = (13.77 * (1.18 * spd**3 / M_BE / constants.G**(3/2))**2).to(u.g/u.cm**3)
    r_0 = (spd / np.sqrt(4 * np.pi * constants.G * rho_c)).to(u.AU)
    return rho_c,r_0

def r_BE_max(M_cl,efficiency,T=10*u.K,R_cl=1*u.pc,mu=2.4*u.Da):
    rho_c,r_0 = BE_props(M_cl,efficiency,T,R_cl,mu)
    return 6.4 * r_0

def density_profile(rr,rho_c,r_0,const=2.24,alpha=2.33):
    return rho_c / (1 + (rr / (const * r_0))**alpha)

def avg_surf_dens(r,M_cl,efficiency,T,R_cl,mu):
    sz = 100
    zz, yy, xx = np.indices([sz, sz, sz])
    rho_c,r_0 = BE_props(M_cl,efficiency,T,R_cl,mu)
    r_max = 6.4 * r_0

    rr_3d = r_max * ((zz - sz / 2.)**2 + (yy - sz / 2.)**2 + (xx - sz / 2.)**2)**0.5 / (sz / 2.)
    rr_2d = r_max * ((yy - sz / 2.)**2 + (xx - sz / 2.)**2)**0.5 / (sz / 2.)

    dens_3d = density_profile(rr_3d,rho_c,r_0)
    dz = 2 * r_max / sz
    column = np.sum(dens_3d * dz,axis=0).to(u.g/u.cm**2)

    return np.mean(column[rr_2d[0] < r])
    
def get_kappa(wav,fn=f'{datapath}/d03_5.5_3.0_A_sub.hdf5',GDR=100):
    d03 = h5py.File(fn,'r+')
    k_d03 = d03['optical_properties']['chi'] * (
        1 - d03['optical_properties']['albedo'])
    d03_nu = d03['optical_properties']['nu'] * u.Hz
    nu = wav.to(u.Hz,equivalencies=u.spectral())
    ret = np.interp(nu[::-1],d03_nu,k_d03) / (GDR + 1)
    return ret[::-1]

#radiation from the prestellar core prior to source ignition
def dust_sphere(M_cl,efficiency,
                wav,aps,distance,
                T=10*u.K,R_cl=1*u.pc,mu=2.4*u.Da):

    M_cl = u.Quantity(M_cl,u.M_sun)
    bb = BlackBody(T)
    kappa = get_kappa(wav) * u.cm**2 / u.g

    r_max = r_BE_max(M_cl,efficiency,T=T,R_cl=R_cl,mu=mu)
    ang_size = (r_max**2 / 4 / distance**2).decompose().value * u.sr

    ret = []
    for ap in aps:
        sigma = avg_surf_dens(ap,M_cl,efficiency,T=T,R_cl=R_cl,mu=mu)
        S = (0.5 * bb(wav) * (1 - np.exp(-sigma * kappa)) * ang_size).to(u.mJy)
        S *= min((ap / r_max)**2,1)
        ret.append(S)
    ret = np.array(ret)
    return ret[None,:,:] * np.ones((9,*ret.shape))
