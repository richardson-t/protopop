import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from abc import ABCMeta

from .helpers import parse_unit_safe,table_to_hdu

class YSOModel(object,metaclass=ABCMeta):

    def __init__(self,
                 distance=None,
                 wav=None,
                 apertures=None,
                 track=None,
                 efficiency=None):

        self.distance = distance
        self.wav = wav
        self.apertures = apertures
        self.track = track
        self.efficiency = efficiency

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self,value):
        if value is None:
            self._distance = None
        else:
            self._distance = value

    @property
    def wav(self):
        return self._wav

    @wav.setter
    def	wav(self,value):
        if value is None:
            self._wav = None
        else:
            self._wav = value

    @property
    def nu(self):
        if self._wav is not None:
            return self._wav.to(u.Hz,equivalencies=u.spectral())
        else:
            return None
            
    @property
    def apertures(self):
        return self._apertures

    @apertures.setter
    def apertures(self,value):
        if value is None:
            self._apertures = None
        else:
            self._apertures = value

    @property
    def track(self):
        return self._track

    @track.setter
    def	track(self,value):
        if value is None:
            self._track = None
        else:
            self._track = value
            
    @classmethod
    def read(cls,filename,order='wav',memmap=True):
        model = cls()

        hdulist = fits.open(filename,memmap=memmap)

        model.distance = hdulist[0].header['DISTANCE'] * u.kpc
        
        hdu_wav = hdulist['WAVELENGTHS']
        model.wav = u.Quantity(hdu_wav.data['WAVELENGTH'],
                               parse_unit_safe(hdu_wav.columns[0].unit))

        hdu_aps = hdulist['APERTURES']
        model.apertures = u.Quantity(hdu_aps.data['APERTURE'],
                                     parse_unit_safe(hdu_aps.columns[0].unit))

        tb = Table()
        hdu_times = hdulist['TIMES']
        tb['Time'] = u.Quantity(hdu_times.data['TIME'],
                                parse_unit_safe(hdu_times.columns[0].unit))
        hdu_vals = hdulist['VALUES']
        vals = hdu_vals.data
        tb.add_column(u.Quantity([v for v in vals],
                                 parse_unit_safe(hdu_vals.header['UNIT'])),
                      name='SED')
        model.track = tb
        
        return model

    def _check_all_set(self):
        if self.distance is None:
            raise ValueError("Value 'distance' is not set")
        if self.wav is None:
            raise ValueError("Wavelengths 'wav' are not set")
        if self.apertures is None:
            raise ValueError("Apertures 'apertures' are not set")
        if self.track is None:
            raise ValueError("Table 'track' is not set")

    def write(self,filename,overwrite=False):
        self._check_all_set()

        hdulist = fits.HDUList()

        hdu0 = fits.PrimaryHDU(data=np.ones(len(self.track)).astype(int))
        hdu0.header['distance'] = (self.distance.value, 'Distance assumed for SED values, in kpc')
        hdu0.header['NWAV'] = (len(self.wav), "Number of wavelengths")
        hdu0.header['NAP'] = (len(self.wav), "Number of apertures")
        hdulist.append(hdu0)

        # Create wavelength table
        t1 = Table()
        t1['WAVELENGTH'] = self.wav
        hdu1 = table_to_hdu(t1)
        hdu1.name = 'WAVELENGTHS'
        hdulist.append(hdu1)

        t2 = Table()
        t2['APERTURE'] = self.apertures
        hdu2 = table_to_hdu(t2)
        hdu2.name = 'APERTURES'
        hdulist.append(hdu2)

        t3 = Table()
        t3['TIME'] = self.track['Time']
        hdu3 = table_to_hdu(t3)
        hdu3.name = 'TIMES'
        hdulist.append(hdu3)

        hdu4 = fits.ImageHDU(self.track['SED'].value)
        hdu4.header['UNIT'] = self.track['SED'].unit.to_string()
        hdu4.name = 'VALUES'
        hdulist.append(hdu4)
        
        hdulist.writeto(filename,overwrite=overwrite)
