import numpy as np
from astropy import units as u
from astropy.table import Table,QTable,vstack
from astropy.io.misc.hdf5 import read_table_hdf5,write_table_hdf5
from imf import make_cluster

from .dust import dust_sphere
from .interpolation import interp_templates
from .time import make_offset
from .util import *

from abc import ABCMeta

from tqdm import tqdm

class Cluster(object,metaclass=ABCMeta):
    """
    Description
    """
    def __init__(self,
                 mass=None,
                 history=None,
                 imf='kroupa',
                 sampling='random',
                 stop_criterion='nearest',
                 sfh=None,
                 timescale=None,
                 efficiency=33,
                 distance=1*u.kpc,
                 T_res=10*u.K,
                 R_res=1*u.pc,
                 mu=2.4*u.Da
    ):
        if mass is not None:
            self._mass = mass

            history_check(history)
            self._history = history

            imf_check(imf)
            self._imf = imf

            if sfh is not None:
                sfh_check(sfh)
                self._sfh = sfh
                if timescale is None:
                    self._timescale = 0.1 * u.Myr
                else:
                    unit_check(timescale,'time')
            else:
                self._sfh = 'start'
                self._timescale = np.nan

            self._sampling = sampling
            self._stop = np.nan if self.sampling == 'optimal' else stop_criterion
            self._efficiency = efficiency
        
            unit_check(distance,'length')
            self._distance = distance
            unit_check(T_res,'temperature')
            unit_check(R_res,'length')
            unit_check(mu,'mass')
            self._res_props = [T_res,R_res,mu]

            self._construct()

    @property
    def mass(self):
        return self._mass

    @property
    def history(self):
        return self._history
    
    @property
    def imf(self):
        return self._imf

    @property
    def sampling(self):
        return self._sampling

    @property
    def stop(self):
        return self._stop

    @property
    def sfh(self):
        return self._sfh

    @property
    def timescale(self):
        return self._timescale

    @property
    def efficiency(self):
        return self._efficiency

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self,value):
        unit_check(value,'length')        
        self._scale_fluxes(value.to(u.kpc))
        self._distance = value.to(u.kpc)
    
    @property
    def wav(self):
        return self._wav

    @property
    def nu(self):
        return self._wav.to(u.Hz,equivalencies=u.spectral())

    @property
    def apertures(self):
        return self._apertures

    @property
    def n_members(self):
        return self._n_members

    @property
    def ev_history(self):
        return self._ev_history

    @property
    def flux_history(self):
        return self._flux_history

    @property
    def inclinations(self):
        return self._inclinations

    @property
    def binaries(self):
        return self._binaries

    @property
    def max_time(self):
        return self._max_time

    @property
    def res_props(self):
        return self._res_props

    def add_time(self,time):
        """
        Description
        """
        unit_check(time,'time')
        time = time.to(u.Myr).value
        
        time = np.atleast_1d(time)
        if not np.logical_or(len(time) == 1,len(time) == self.n_members):
            raise ValueError("'time' must either be a single value or match number of cluster members")
        if len(time) == 1:
            for tbl in self.ev_history.values():
                tbl['Time'] += time
            for	tbl in self.flux_history.values():
                tbl['Time'] += time
        else:
            for i,tbl in enumerate(self.ev_history.values()):
                tbl['Time'] += time[i]
            for i,tbl in enumerate(self.flux_history.values()):
                tbl['Time'] += time[i]
    
    #Find earliest start time in an evolutionary track
    def _min_time(self):
        times = [np.min(tbl['Time']) for tbl in self.ev_history.values()]
        mintime = np.min(times)
        return mintime
            
    #Align evolutionary tracks such that accretion ends at the same time
    def _align_end(self):
        times = np.array([tbl['Time'][-1] for tbl in self.flux_history.values()])
        add_times = np.max(times) - times
        self.add_time(add_times)

    #Scale flux values from their current distance to a new distance
    def _scale_fluxes(self,d_new):
        factor = (self.distance**2 / d_new**2).value
        for tbl in self.flux_history.values():
            tbl['SED'] *= factor

    #Set up the cluster
    def _construct(self):
        print('Setting up...')
        info = setup_templates(self.history,
                               self.efficiency)

        print('Calculating initial conditions...')
        mass_key = [*info[1][0].keys()][0]
        distance = info[1][1][mass_key].distance
        self._wav = info[1][1][mass_key].wav
        self._apertures = info[1][1][mass_key].apertures
        inits = dust_sphere(self.mass,self.efficiency,
                            self.wav,self.apertures,
                            distance,
                            T=self.res_props[0],
                            R_cl=self.res_props[1],
                            mu=self.res_props[2])
        bounds = (inits,distance,
                  self._wav,self._apertures)

        print('Sampling members...')
        masses = make_cluster(self.mass,
                              massfunc=self.imf,
                              mmin=0.03,
                              mmax=120,
                              sampling=self.sampling,
                              stop_criterion=self.stop)
        masses = masses[np.argsort(masses)]
        #restrict modeled stars to ones with final mass > 0.2 Msun
        masses = masses[masses >= 0.2]

        inclinations = pick_inclinations(masses)
        isBinary = pick_binaries(masses)

        #restrict modeled binaries to ones with total mass > 0.4 Msun
        bin_cut = np.logical_and(isBinary,masses < 0.4)
        masses = masses[~bin_cut]
        self._binaries = isBinary[~bin_cut]
        self._inclinations = inclinations[~bin_cut]
        self._n_members = len(masses)
        
        print(f'Calculating histories for {len(masses)} systems...')
        ev_history = {}
        flux_history = {}
        for i,m in tqdm(enumerate(masses)):
            track_info = info[2] if self.binaries[i] else info[1]
            evol,flux = interp_templates(m,info[0],*track_info,*info[-2:],*bounds)
            ev_history.update({m:evol})
            flux_history.update({m:flux})

        self._ev_history = ev_history
        self._flux_history = flux_history
        self._scale_fluxes(self.distance)
        
        if self.sfh is not None:
            if 'end' in self.sfh:
                self.align_end()
            if self.sfh not in ['start','end']:
                offset_times = make_offset(self.n_members,self.sfh,self.timescale)
                self.add_time(offset_times)
        
        times = []
        for tbl in self.flux_history.values():
            times.append(tbl['Time'][-1])
        self._max_time = np.max(times) * u.Myr

    def sample_ev(self,time):
        """
        Description
        """
        unit_check(time,'time')
        time = time.to(u.Myr).value
        
        ret = Table()
        for tbl in self.ev_history.values():
            time_index = np.argmin(abs(time - tbl['Time']))
            row = tbl[time_index]
            ret = vstack([ret,row])
        return ret

    #add support for different viewing angles + check if this works for multiple frequencies/apertures simultaneously?
    def sample_flux(self,time,wavelength=None,frequency=None,aperture=1000*u.AU):
        """
        Description
        """
        unit_check(time,'time')
        time = time.to(u.Myr)
        
        if np.logical_or(wavelength is None and frequency is None,
                         wavelength is not None and frequency is not None):
            raise RuntimeError('Provide either wavelength or frequency (in astropy units)')
        if wavelength is not None:
            unit_check(wavelength,'length')
            wavelength = wavelength.to(u.um)
        elif frequency is not None:
            unit_check(frequency,'frequency')
            wavelength = frequency.to(u.um,equivalencies=u.spectral())
        
        ap_index = np.searchsorted(self.apertures,aperture)
        ap_frac = (aperture - self.apertures[ap_index-1]) / (self.apertures[ap_index] - self.apertures[ap_index-1])

        wav_index = np.searchsorted(self.wav,wavelength)
        wav_frac = (wavelength - self.wav[wav_index - 1]) / (self.wav[wav_index] - self.wav[wav_index - 1])

        fluxes = []
        inc_bins = np.linspace(0,90,10)
        for n,tbl in enumerate(self.flux_history.values()):
            inc = self.inclinations[n]
            inc_index = np.searchsorted(inc_bins,inc) - 1
            time_index = np.argmin(abs(time - tbl['Time']))
            row = tbl[time_index]
            row_sed = row['SED'][inc_index]
            ap_interp = (1. - ap_frac) * row_sed[ap_index-1] + ap_frac * row_sed[ap_index]
            wav_interp = (1. - wav_frac) * ap_interp[wav_index-1] + wav_frac * ap_interp[wav_index]
            fluxes.append(wav_interp)

        return np.array(fluxes)

    @classmethod
    def read(cls,filename):
        cluster = cls()
        
        in_file = h5py.File(filename,'r')
        
        prop_table = read_table_hdf5(in_file,path='properties')
        cluster._mass = prop_table['mass'][0]
        cluster._history = prop_table['history'][0]
        cluster._imf = prop_table['imf'][0]
        cluster._sampling = prop_table['sampling'][0]
        cluster._stop = prop_table['stop'][0]
        cluster._sfh = prop_table['sfh'][0]
        cluster._timescale = prop_table['timescale'][0]
        cluster._efficiency = prop_table['efficiency'][0]
        cluster._distance = prop_table['distance'][0]
        cluster._wav = prop_table['wav'][0]
        cluster._apertures = prop_table['ap'][0]
        cluster._res_props = [prop_table['tres'][0],
                              prop_table['rres'][0],
                              prop_table['mu'][0]]
        cluster._n_members = prop_table['n_members'][0]
        cluster._inclinations = prop_table['inclinations'][0]
        cluster._binaries = prop_table['binaries'][0]
        cluster._max_time = prop_table['max_time'][0]

        masses = []
        for k in in_file['ev'].keys():
            masses.append(np.float64(k.split('/')[-1]))
        masses.sort()
            
        ev_hist = dict()
        flux_hist = dict()
        for m in tqdm(masses,desc='Reading tracks',ncols=0,leave=False):
            ev_hist.update({m:read_table_hdf5(in_file,path=f'ev/{m}')})
            flux_hist.update({m:read_table_hdf5(in_file,path=f'flux/{m}')})
        cluster._ev_history = ev_hist
        cluster._flux_history = flux_hist
        in_file.close()

        return cluster
        
    def write(self,filename,overwrite=True):
        if overwrite:
            open_str = 'w'
        else:
            open_str = 'x'
            
        out_file = h5py.File(f'{filename}.hdf5',open_str)

        prop_table = QTable()
        prop_table.add_column([self.mass],name='mass')
        prop_table.add_column([self.history],name='history')
        prop_table.add_column([self.imf],name='imf')
        prop_table.add_column([self.sampling],name='sampling')
        prop_table.add_column([self.stop],name='stop')
        prop_table.add_column([self.sfh],name='sfh')
        prop_table.add_column([self.timescale],name='timescale')
        prop_table.add_column([self.efficiency],name='efficiency')
        prop_table.add_column([self.distance],name='distance')
        prop_table.add_column([self.wav],name='wav')
        prop_table.add_column([self.apertures],name='ap')
        prop_table.add_column([self.res_props[0]],name='tres')
        prop_table.add_column([self.res_props[1]],name='rres')
        prop_table.add_column([self.res_props[2]],name='mu')
        prop_table.add_column([self.n_members],name='n_members')
        prop_table.add_column([self.inclinations],name='inclinations')
        prop_table.add_column([self.binaries],name='binaries')
        prop_table.add_column([self.max_time],name='max_time')
        write_table_hdf5(prop_table,out_file,path='properties',
                         compression=True,serialize_meta=True)

        for key in tqdm(self.ev_history.keys(),desc='Writing tracks',ncols=0,leave=False):
            write_table_hdf5(self.ev_history[key],out_file,path=f'ev/{key}',
                             compression=True)
            write_table_hdf5(self.flux_history[key],out_file,path=f'flux/{key}',
                             compression=True,serialize_meta=True)
        
        out_file.close()
