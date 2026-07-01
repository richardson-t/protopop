import numpy as np
from astropy import units as u
from astropy.table import Table
from astroquery.svo_fps import SvoFps
import h5py

from ..yso import YSOModel

from os import path
from glob import glob

datapath = path.dirname(__file__) + '/..'

# Create dicts used in cluster sampling operations


def setup_templates(history, efficiency):
    history_check(history)

    ev_files = glob(f'{datapath}/track_data/protostar_tracks/{history}/*.txt')
    masses = np.array([float(f.split('=')[-1].split('.')[0] + '.' +
                      f.split('=')[-1].split('.')[1]) for f in ev_files])
    indices = np.argsort(masses)
    masses = masses[indices]

    ev_tracks = {masses[pair[0]]: Table.read(ev_files[pair[1]], format='ascii')
                 for pair in enumerate(indices)}
    flux_tracks = {masses[i]: YSOModel.read(f'{datapath}/track_data/flux_tracks/{history}/mf={masses[i]}_eff={efficiency}.fits')
                   for i in range(len(masses))}
    b_ev_tracks = {masses[i+1]: Table.read(f'{datapath}/track_data/protostar_tracks/{history}/binaries/mf={masses[i+1]}.txt', format='ascii')
                   for i in range(len(masses[1:]))}
    b_flux_tracks = {masses[i+1]: YSOModel.read(f'{datapath}/track_data/flux_tracks/{history}/binaries/mf={masses[i+1]}_eff={efficiency}.fits')
                     for i in range(len(masses[1:]))}

    # first timestep + temperature where the star stops accreting
    last_times = {mass: tbl['Time'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0]
                  for mass, tbl in ev_tracks.items()}
    last_temps = {mass: tbl['Stellar_Temperature'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0]
                  for mass, tbl in ev_tracks.items()}
    b_last_temps = {mass: tbl['Stellar_Temperature'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0]
                    for mass, tbl in b_ev_tracks.items()}

    single_info = (ev_tracks, flux_tracks, last_temps)
    binary_info = (b_ev_tracks, b_flux_tracks, b_last_temps)

    return masses, single_info, binary_info, last_times, history


def pick_inclinations(vals):
    """
    Draw :math:`n` inclinations between :math:`0-90^{\\circ}`, where
    :math:`n` is the length of list/array-like ``vals``
    """
    rng = np.random.default_rng()
    inclinations = rng.random(len(vals)) * 90
    return inclinations

# somewhat redundant with imf, but this is limited to binaries


def pick_binaries(syst_masses):
    """
    For an array of star system masses, determine probability of
    being a binary based on the multiplicity fractions of
    `Offner et al. (2023) <https://doi.org/10.48550/arXiv.2203.10066>`_. 
    Returns an array of booleans of the same length as the 
    input array; ``True`` entries are binaries.
    """
    mults = Table.read(f'{datapath}/track_data/multiplicity.fits')
    rng = np.random.default_rng()
    probs = rng.random(len(syst_masses))
    fractions = np.interp(syst_masses, mults['Primary Mass'], mults['Multiplicity Fraction'])
    return np.logical_and(probs < fractions, syst_masses > 0.4)

# return the location/fractional position between entries of a value
# in an array (not in use/deprecated)?


def interp_props(x, base_x):
    x = np.atleast_1d(x)
    returnScalar = True if len(x) == 1 else False

    indices = np.searchsorted(base_x, x)
    fracs = []
    for i, val in enumerate(x):
        x1 = base_x[indices[i]-1]
        x2 = base_x[indices[i]]
        fracs.append((val - x1) / (x2 - x1))
    fracs = np.array(fracs)
    if returnScalar:
        return indices[0], fracs[0]
    else:
        return indices, fracs


def filter_flux(sed, wav, instrument, camera, returnZero=True):
    """
    Convolve an SED with an instrumental response profile. Profiles
    are pulled from the Spanish Virtual Observatory's
    `Filter Profile Service <https://svo2.cab.inta-csic.es/theory/fps/>`_
    (SVO FPS) via `astroquery <https://astroquery.readthedocs.io/en/latest/>`_.

    Parameters
    ----------
    sed: array
        The SED to be convolved
    wav: :math:`{\\rm \\mu m}` or equivalent
        Wavelength(s) where the SED to be convolved is defined
    instrument: str
        The instrument with the filter, formatted as in the SVO FPS
        (e.g. `JWST`)
    camera: str
        The actual filter to be convolved, formatted as in the SVO FPS
        (e.g. `MIRI.F2550W`)

    Other Parameters
    ----------------
    returnZero: bool
        If ``True``, returns the Vega zero point (in Jy) for the filter

    Returns
    -------
    flux: :math:`{\\rm mJy}`
        Convolved flux
    zeropoint: :math:`{\\rm Jy}`, optional
        Zero point of the filter
    """
    filter_info = SvoFps.get_transmission_data(f'{instrument}/{camera}')
    filter_wav = (filter_info['Wavelength']).to(u.um)
    filter_response = filter_info['Transmission']
    interp_flux = np.interp(filter_wav, wav, sed)
    avresponse = (filter_response[:-1] + filter_response[1:])/2
    vals = interp_flux * filter_response
    vals = (vals[:1] + vals[:-1]) / 2
    dlambda = filter_wav[1:] - filter_wav[:-1]
    flux = np.sum(vals * dlambda) / np.sum(avresponse * dlambda * u.um)
    if returnZero:
        table = SvoFps.get_filter_list(instrument)
        zeropoint = table['ZeroPoint'][table['filterID'] == f'{instrument}/{camera}'][0] * u.Jy
        return flux, zeropoint
    else:
        return flux

# Round a number to some number of significant figures


def sig_round(number, sigfigs=3):
    if number == 0:
        return 0
    else:
        mag = np.floor(np.log10(number)).astype(int)
        return np.round(number, sigfigs - mag)

# Retrieve a particular value from an array entry in an astropy table
# review this; probably belongs in a track-making code instead of here


def get_mass(table, row, ap, key='Sphere Masses'):
    return np.nanmax(table[key][row, 0, :ap+1])

# check that an entered IMF model is supported


def imf_check(imf):
    approved_imfs = ['kroupa', 'chabrier', 'salpeter']
    if imf in approved_imfs:
        pass
    else:
        raise ValueError('IMF not recognized')

# check that an entered accretion history is supported


def history_check(history):
    approved_histories = ['is', 'tc', 'ca', 'exp',
                          'taper_is', 'taper_tc', 'taper_ca']
    if history in approved_histories:
        pass
    else:
        raise ValueError('Accretion history not recognized')

# check that an entered SFH is supported


def sfh_check(sfh):
    approved_sfhs = ['start', 'end', 'constant', 'normalstart', 'normalend']
    if sfh in approved_sfhs:
        pass
    else:
        raise ValueError('Star formation history not recognized')

# check that a value is the expected type of astropy unit


def unit_check(val, expected_unit):
    try:
        assert u.get_physical_type(val) == expected_unit
    except (AssertionError):
        raise TypeError(f'argument must be in astropy {expected_unit} units')
