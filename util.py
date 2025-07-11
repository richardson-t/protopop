import numpy as np
from astropy import units as u
from astropy.table import Table
from astroquery.svo_fps import SvoFps
import h5py

from .yso import YSOModel

from glob import glob
import os

def setup_templates(history,efficiency):
    history_check(history)

    datapath = os.path.dirname(__file__)

    ev_files = glob(f'{datapath}/data/protostar_tracks/{history}/*.txt')
    masses = np.array([float(f.split('=')[-1].split('.')[0] + '.' + f.split('=')[-1].split('.')[1]) for f in ev_files])
    indices = np.argsort(masses)
    masses = masses[indices]
    
    ev_tracks = {masses[pair[0]]:Table.read(ev_files[pair[1]],format='ascii')
                 for pair in enumerate(indices)}
    flux_tracks = {masses[i]:YSOModel.read(f'{datapath}/data/flux_tracks/{history}/mf={masses[i]}_eff={efficiency}.fits')
                   for i in range(len(masses))}
    b_ev_tracks = {masses[i+1]:Table.read(f'{datapath}/data/protostar_tracks/{history}/binaries/mf={masses[i+1]}.txt',format='ascii')
                   for i in range(len(masses[1:]))}
    b_flux_tracks = {masses[i+1]:YSOModel.read(f'{datapath}/data/flux_tracks/{history}/binaries/mf={masses[i+1]}_eff={efficiency}.fits')
                     for i in range(len(masses[1:]))}
    
    # first timestep + temperature where the star stops accreting
    last_times = {mass: tbl['Time'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0]
                  for mass,tbl in ev_tracks.items()}
    last_temps = {mass: tbl['Stellar_Temperature'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0]
                  for mass,tbl in ev_tracks.items()}
    b_last_temps = {mass: tbl['Stellar_Temperature'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0]
                    for mass,tbl in b_ev_tracks.items()}

    single_info = (ev_tracks,flux_tracks,last_temps)
    binary_info = (b_ev_tracks,b_flux_tracks,b_last_temps)
    
    return masses,single_info,binary_info,last_times,history

def pick_inclinations(vals):
    rng = np.random.default_rng()
    inclinations = rng.random(len(vals)) * 90
    return inclinations

def pick_binaries(syst_masses):
    datapath = os.path.dirname(__file__)
    
    mults = Table.read(f'{datapath}/data/multiplicity.fits')
    rng = np.random.default_rng()
    probs = rng.random(len(syst_masses))
    fractions = np.interp(syst_masses,mults['Primary Mass'],mults['Multiplicity Fraction'])
    return np.logical_or(probs < fractions,syst_masses < 0.4)

def interp_props(x,base_x):
    x = np.atleast_1d(x)
    returnScalar = True if len(x) == 1 else False
    
    indices = np.searchsorted(base_x,x)
    fracs = []
    for i,val in enumerate(x):
        x1 = base_x[indices[i-1]]
        x2 = base_x[indices[i]]
        fracs.append((val - x1) / (x2 - x1))
    fracs = np.array(fracs)
    if returnScalar:
        return indices[0],fracs[0]
    else:
        return indices,fracs

def filter_flux(wav,sed,instrument,camera,returnZero=True):
    filter_info = SvoFps.get_transmission_data(f'{instrument}/{camera}')
    filter_wav = (filter_info['Wavelength']).to(u.um)
    filter_response = filter_info['Transmission']
    interp_flux = np.interp(filter_wav,wav,sed)
    avresponse = (filter_response[:-1] + filter_response[1:])/2
    vals = interp_flux * filter_response
    vals = (vals[:1] + vals[:-1]) / 2
    dlambda = filter_wav[1:] - filter_wav[:-1]
    flux = np.sum(vals * dlambda) / np.sum(avresponse * dlambda * u.um)
    if returnZero:
        table = SvoFps.get_filter_list(instrument)
        zeropoint = table['ZeroPoint'][table['filterID'] == f'{instrument}/{camera}'][0] * u.Jy
        return flux,zeropoint
    else:
        return flux

def sig_round(number,sigfigs=3):
    if number == 0:
        return 0
    else:
        mag = np.floor(np.log10(number)).astype(int)
        return np.round(number,sigfigs - mag)

def get_mass(table,row,ap,key):
    return np.nanmax(table[key][row,0,:ap+1])

def imf_check(imf):
    approved_imfs = ['kroupa','chabrier','salpeter']
    if imf in approved_imfs:
        pass
    else:
        raise ValueError('IMF not recognized')

def history_check(history):
    approved_histories = ['is','tc','ca','exp',
                          'taper_is','taper_tc','taper_ca']
    if history in approved_histories:
        pass
    else:
        raise ValueError('Accretion history not recognized')

def sfh_check(sfh):
    approved_sfhs = ['start','end','constant','normalstart','normalend']
    if sfh in approved_sfhs:
        pass
    else:
        raise ValueError('Star formation history not recognized')

def unit_check(val,expected_unit):
    try:
        assert u.get_physical_type(val) == expected_unit
    except (AssertionError):
        raise TypeError(f'argument must be in astropy {expected_unit} units')

