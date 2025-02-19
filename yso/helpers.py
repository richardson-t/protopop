import numpy as np
from astropy import units as u
from astropy.io import fits

UNIT_MAPPING = {}
UNIT_MAPPING['MICRONS'] = u.um
UNIT_MAPPING['HZ'] = u.Hz
UNIT_MAPPING['MJY'] = u.mJy
UNIT_MAPPING['ergs/cm^2/s'] = u.erg / u.cm ** 2 / u.s

def parse_unit_safe(unit_string):
    if unit_string in UNIT_MAPPING:
        return UNIT_MAPPING[unit_string]
    else:
        return u.Unit(unit_string, parse_strict=False)

def table_to_hdu(table):
    hdu = fits.BinTableHDU(np.array(table))
    for i in range(len(table.columns)):
        if table.columns[i].unit is not None:
            hdu.columns[i].unit = table.columns[i].unit.to_string(format='fits')
    return hdu
