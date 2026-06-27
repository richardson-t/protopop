import numpy as np
from astropy import units as u
from ..util import sfh_check, unit_check

# Generate n random numbers from a uniform distribution
# over 0 to maxtime


def _make_random(n, max_val):
    rng = np.random.default_rng()
    return u.Quantity(rng.uniform(0, max_val.value, n), max_val.unit)

# Generate n random numbers from a normal distribution
# centered on 0 with a 1-sigma of interval


def _make_normal_random(n, sigma):
    rng = np.random.default_rng()
    return u.Quantity(rng.normal(0, sigma.value, n), sigma.unit)


def make_offset(sfh, n_times, timescale):
    """
    Sample start times for protocluster members from some
    star formation history.

    Parameters
    ----------
    sfh: str
        Star formation history to sample from. Accepts "constant" (uniform
        distribution) and "normalstart"/"normalend" (normal distribution)
    n_times: int
        Number of times to sample
    timescale: :math:`{\\rm Myr}` or equivalent
        Characteristic timescale for the distribution. "constant" = the 
        total timespan over which to sample; "normal" = the 1-sigma width
        of the distribution
    """
    sfh_check(sfh)
    unit_check(timescale, 'time')
    if sfh == 'constant':
        ret = make_random(n_times, timescale)
    else:
        ret = make_normal_random(n_times, timescale)

    ret -= min(ret)
    return ret
