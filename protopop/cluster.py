import numpy as np
from astropy import units as u
from astropy.table import Table, QTable, vstack
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
from scipy.interpolate import RegularGridInterpolator
from imf.sampling import sample_mass

from .interpolation import interp_tracks
from .time import make_offset
from .util import *

from abc import ABCMeta

from tqdm import tqdm


class Cluster(object, metaclass=ABCMeta):
    """
    A model protocluster. For details on the input options 
    supported by this version of protopop, see the 
    "Configuration" page of the documentation. 

    Parameters
    ----------
    mass: float
        Total mass in stars of the final stellar population,
        in Msun
    imf: str, MassFunction
        Mass function from which the final stellar population
        is drawn
    sampling: str
        Sampling method used to draw masses from the IMF
    stop_criterion: str
        Which criterion to use to stop sampling (if sampling
        is done randomly)
    history: str
        Accretion history followed by the model YSOs making
        up the protocluster
    sfh: str
        History of star formation followed by the cluster members
    timescale: :math:`{\\rm Myr}` or equivalent
        Characteristic timescale for the star formation history
    efficiency: float
        Mass accretion efficiency for star formation, as a percentage
        (i.e. 33 -> 33% of mass transferred from natal material to
        star)
    distance: :math:`{\\rm kpc}` or equivalent
        Distance to the cluster
    T_res: :math:`{\\rm K}` or equivalent
        Average temperature of the material in the
        cluster's parent mass reservoir prior to
        any formation (default = 10 K)
    R_res: :math:`{\\rm pc}` or equivalent
        Size of the parent mass reservoir; assumed
        to be a sphere (default = 1 pc)
    mu: :math:`{\\rm Da}` or equivalent
        Mean molecular weight of the material in the
        parent mass reservoir (default = 2.4 Da)

    Other Parameters                       
    ----------------
    read_only: bool
        If ``True``, do not assign anything to
        cluster attributes on instantiation. Used
        for reading existing protocluster models.
    """

    def __init__(self,
                 mass=None,
                 imf='kroupa',
                 sampling='random',
                 stop_criterion='nearest',
                 history=None,
                 sfh=None,
                 timescale=None,
                 efficiency=33,
                 distance=1*u.kpc,
                 T_res=10*u.K,
                 R_res=1*u.pc,
                 mu=2.4*u.Da,
                 read_only=False
                 ):
        if not read_only:
            if mass is None:
                raise ValueError('Total cluster mass must be provided')
            self._mass = mass

            imf_check(imf)
            self._imf = imf

            history_check(history)
            self._history = history

            if sfh is not None:
                sfh_check(sfh)
                self._sfh = sfh
                if timescale is None:
                    self._timescale = 0.1 * u.Myr
                else:
                    unit_check(timescale, 'time')
            else:
                self._sfh = 'start'
                self._timescale = np.nan

            self._sampling = sampling
            self._stop = np.nan if self.sampling == 'optimal' else stop_criterion
            self._efficiency = efficiency

            unit_check(distance, 'length')
            self._distance = distance
            unit_check(T_res, 'temperature')
            unit_check(R_res, 'length')
            unit_check(mu, 'mass')
            self._res_props = [T_res, R_res, mu]

            self._construct()

    @property
    def mass(self):
        return self._mass

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
    def history(self):
        return self._history

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
    def distance(self, value):
        unit_check(value, 'length')
        self._distance = value.to(u.kpc)

    @property
    def wav(self):
        """
        Wavelength(s) at which the model YSO SEDs are defined
        (in :math:`{\\rm \\mu m}`)
        """
        return self._wav

    @property
    def nu(self):
        """
        Frequency(/ies) at which the model YSO SEDs are defined 
        (in :math:`{\\rm Hz}`)
        """
        return self._wav.to(u.Hz, equivalencies=u.spectral())

    @property
    def apertures(self):
        """
        Apertures in which the model YSO SEDs are defined
        (in :math:`{\\rm AU}`)
        """
        return self._apertures

    @property
    def n_members(self):
        """
        Number of cluster members
        """
        return self._n_members

    @property
    def member_masses(self):
        """
        Final masses of cluster members
        """
        return self._member_masses

    @property
    def inclinations(self):
        """
        Viewing angles of cluster members
        """
        return self._inclinations

    @property
    def binaries(self):
        """
        Whether or not cluster members are binary systems
        """
        return self._binaries

    @property
    def end_times(self):
        """
        When each member stops accreting (in :math:`{\\rm Myr}`)
        """
        return self._end_times

    @property
    def max_time(self):
        """
        When the last cluster member stops accreting
        (in :math:`{\\rm Myr}`)
        """
        return self._max_time

    @property
    def offsets(self):
        """
        When each cluster member starts accreting
        (in :math:`{\\rm Myr}`)
        """
        return self._offsets

    @property
    def res_props(self):
        """
        Initial properties (temperature, radius, molecular weight)
        of the material in the cluster's parent mass reservoir
        """
        return self._res_props

    def add_time(self, time):
        """
        Add (or subtract) time from the timelines of
        protocluster members. Accepts a single value or
        an array of values with length equal to the number of
        cluster members (must have astropy time units). Single
        values will be added to all timelines; values in arrays
        will be added to their corresponding member.
        """
        unit_check(time, 'time')
        time = time.to(u.Myr)

        time = np.atleast_1d(time)
        if not np.logical_or(len(time) == 1, len(time) == self.n_members):
            raise ValueError(
                "'time' must either be a single value or match number of cluster members")

        self._end_times += time
        self._offsets += time
        self._max_time = np.max(self._end_times.value) * u.Myr

    # Align evolutionary tracks such that accretion ends at the same time
    def _align_end(self):
        add_times = self.max_time - self.end_times
        self.add_time(add_times)

    # Set up the cluster
    def _construct(self):
        print('Setting up...')
        self._info = setup_templates(self.history,
                                     self.efficiency)

        print('Calculating initial conditions...')
        mass_key = [*self._info[1][0].keys()][0]
        self._track_distance = self._info[1][1][mass_key].distance
        self._wav = self._info[1][1][mass_key].wav
        self._apertures = self._info[1][1][mass_key].apertures
        del mass_key

        print('Sampling members...')
        masses = sample_mass(self.mass,
                             massfunc=self.imf,
                             mmin=0.03,
                             mmax=120,
                             sampling=self.sampling,
                             stop_criterion=self.stop)
        self._mass = sum(masses)
        masses = masses[masses > 0.2]
        self._n_members = len(masses)
        self._member_masses = np.copy(masses[np.argsort(masses)])
        del masses

        self._inclinations = pick_inclinations(self.member_masses)
        self._binaries = pick_binaries(self.member_masses)

        indices, fracs = interp_props(self.member_masses, self._info[0])
        end_times = []
        for i in range(len(indices)):
            t1 = self._info[1][0][self._info[0][indices[i]-1]]['Time'][-1]
            t2 = self._info[1][0][self._info[0][indices[i]]]['Time'][-1]
            t = (1. - fracs[i]) * t1 + fracs[i] * t2
            end_times.append(t)
        self._end_times = np.array(end_times) * u.Myr
        self._max_time = np.max(self.end_times)
        self._offsets = np.zeros(len(self.member_masses)) * u.Myr
        del indices, fracs

        if self.sfh is not None:
            if 'end' in self.sfh:
                self.align_end()
            if self.sfh not in ['start', 'end']:
                offset_times = make_offset(self.n_members, self.sfh, self.timescale)
                self.add_time(offset_times)

    def sample_ev(self, time):
        """
        Sample the evolutionary history of protocluster
        members at a particular time (must have astropy
        time units). Returns a table where each row is
        output from the protostellar evolutionary track
        underlying a member; values are calculated through
        interpolation of tracks created with a modified
        `Klassen+ (2012) <https://doi.org/10.1111/j.1365-2966.2012.20523.x>`__ 
        code (see `Richardson+ 2025 <https://doi.org/10.3847/1538-4357/ade99d>`__).
        """
        unit_check(time, 'time')
        time = time.to(u.Myr).value

        ret = Table()
        for i, m in tqdm(enumerate(self.member_masses),
                         total=self.n_members, leave=False):
            track_index = 2 if self.binaries[i] else 1
            evol = interp_tracks(m,
                                 self._info[0], *self._info[track_index], *self._info[-2:],
                                 self._track_distance, self.wav, self.apertures,
                                 self.mass, self.efficiency, self.res_props,
                                 return_ev=True)
            evol['Time'] += self.offsets[i].value
            time_index = np.argmin(abs(time - evol['Time']))
            row = evol[time_index]
            ret = vstack([ret, row])
        return ret

    # add support for different viewing angles?
    def sample_flux(self, time, wav=None, freq=None, ap=1000*u.AU):
        """
        Sample the flux evolution of cluster members. Returns
        an array of flux values (in mJy) corresponding to the
        flux predicted for each member at the requested time.
        Flux evolution is calculated following the procedure
        laid out in `Richardson+ (2025) <https://doi.org/10.3847/1538-4357/ade99d>`__.

        Parameters
        ----------
        time: :math:`{\\rm Myr}` or equivalent
            Time at which to sample the cluster
        wav: :math:`{\\rm \\mu m}` or equivalent
            Wavelength(s) at which to sample the cluster.
            Degenerate with ``freq`` (default = ``None``)
        freq: :math:`{\\rm Hz}` or equivalent
            Frequency(/ies) at which to sample the cluster.
            Degenerate with ``wav`` (default = ``None``)
        ap: :math:`{\\rm AU}` or equivalent
            Aperture(s) at which to sample the cluster
            (default = 1000 AU)
        """
        unit_check(time, 'time')
        time = time.to(u.Myr).value

        if np.logical_or(wav is None and freq is None,
                         wav is not None and freq is not None):
            raise RuntimeError('Provide either wavelength or frequency (in astropy units)')
        if wav is not None:
            unit_check(wav, 'length')
            wav = wav.to(u.um)
        elif freq is not None:
            unit_check(freq, 'frequency')
            wav = freq.to(u.um, equivalencies=u.spectral())
        wav = np.atleast_1d(wav.value) * u.um
        scalarWav = True if len(wav) == 1 else False

        unit_check(ap, 'length')
        ap = ap.to(u.AU)
        ap = np.atleast_1d(ap.value) * u.AU
        scalarAp = True if len(ap) == 1 else False

        fluxes = []
        inc_bins = np.linspace(0, 90, 10)
        for i, m in tqdm(enumerate(self.member_masses),
                         total=self.n_members, leave=False):
            track_index = 2 if self.binaries[i] else 1
            flux = interp_tracks(m,
                                 self._info[0], *self._info[track_index], *self._info[-2:],
                                 self._track_distance, self.wav, self.apertures,
                                 self.mass, self.efficiency, self.res_props,
                                 return_flux=True)
            flux['Time'] += self.offsets[i].value
            row = flux[np.argmin(abs(time - flux['Time']))]

            inc = self.inclinations[i]
            inc_index = np.searchsorted(inc_bins, inc) - 1
            row_sed = row['SED'][inc_index]
            try:
                unit_check(row_sed, u.mJy)
            except (TypeError):
                row_sed = row_sed * u.mJy
            row_sed *= (self._track_distance / self.distance)**2
            grid = RegularGridInterpolator((self.apertures.value, self.wav.value), row_sed)
            aa, ww = np.meshgrid(ap.value, wav.value, indexing='ij')
            ret = grid((aa, ww))
            if scalarAp and scalarWav:
                ret = ret[0, 0]
            elif scalarAp:
                ret = ret[0]
            elif scalarWav:
                ret = ret[:, 0]
            fluxes.append(ret)

        return np.array(fluxes)

    def sample_all(self, time, wav=None, freq=None, ap=1000*u.AU):
        """
        Sample both the evolutionary tracks and flux tracks
        of cluster members.

        Returns
        -------
        ev: astropy table
        fluxes: array
        """
        unit_check(time, 'time')
        time = time.to(u.Myr).value

        if np.logical_or(wav is None and freq is None,
                         wav is not None and freq is not None):
            raise RuntimeError('Provide either wavelength or frequency (in astropy units)')
        if wav is not None:
            unit_check(wav, 'length')
            wav = wav.to(u.um)
        elif freq is not None:
            unit_check(freq, 'frequency')
            wav = freq.to(u.um, equivalencies=u.spectral())
        wav = np.atleast_1d(wav.value) * u.um
        scalarWav = True if len(wav) == 1 else False

        unit_check(ap, 'length')
        ap = ap.to(u.AU)
        ap = np.atleast_1d(ap.value) * u.AU
        scalarAp = True if len(ap) == 1 else False

        ev = Table()
        fluxes = []
        inc_bins = np.linspace(0, 90, 10)
        for i, m in tqdm(enumerate(self.member_masses),
                         total=self.n_members, leave=False):
            track_index = 2 if self.binaries[i] else 1
            m_evol, m_flux = interp_tracks(m,
                                           self._info[0], *self._info[track_index], *
                                           self._info[-2:],
                                           self._track_distance, self.wav, self.apertures,
                                           self.mass, self.efficiency, self.res_props,
                                           return_ev=True, return_flux=True)

            m_evol['Time'] += self.offsets[i].value
            time_index = np.argmin(abs(time - m_evol['Time']))
            row = m_evol[time_index]
            ev = vstack([ev, row])

            m_flux['Time'] += self.offsets[i].value
            row = m_flux[np.argmin(abs(time - m_flux['Time']))]

            inc = self.inclinations[i]
            inc_index = np.searchsorted(inc_bins, inc) - 1
            row_sed = row['SED'][inc_index]
            try:
                unit_check(row_sed, u.mJy)
            except (TypeError):
                row_sed = row_sed * u.mJy
            row_sed *= (self._track_distance / self.distance)**2
            grid = RegularGridInterpolator((self.apertures.value, self.wav.value), row_sed)
            aa, ww = np.meshgrid(ap.value, wav.value, indexing='ij')
            val = grid((aa, ww))
            if scalarAp and scalarWav:
                val = val[0, 0]
            elif scalarAp:
                val = val[0]
            elif scalarWav:
                val = val[:, 0]
            fluxes.append(val)

        return ev, np.array(fluxes)

    @classmethod
    def read(cls, filename):
        cluster = cls(read_only=True)

        print('Reading cluster properties...')
        in_file = h5py.File(filename, 'r')

        prop_table = read_table_hdf5(in_file, path='properties')
        cluster._mass = prop_table['mass'][0]
        cluster._imf = prop_table['imf'][0]
        cluster._sampling = prop_table['sampling'][0]
        cluster._stop = prop_table['stop'][0]
        cluster._history = prop_table['history'][0]
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
        cluster._member_masses = prop_table['member_masses'][0]
        cluster._inclinations = prop_table['inclinations'][0]
        cluster._binaries = prop_table['binaries'][0]
        cluster._end_times = prop_table['end_times'][0]
        cluster._max_time = prop_table['max_time'][0]
        cluster._offsets = prop_table['offsets'][0]

        in_file.close()

        print('Final setup...')
        cluster._info = setup_templates(cluster.history,
                                        cluster.efficiency)
        mass_key = [*cluster._info[1][0].keys()][0]
        cluster._track_distance = cluster._info[1][1][mass_key].distance
        del mass_key

        return cluster

    def write(self, filename, overwrite=True):
        if overwrite:
            open_str = 'w'
        else:
            open_str = 'x'

        prop_table = QTable()
        prop_table.add_column([self.mass], name='mass')
        prop_table.add_column([self.imf], name='imf')
        prop_table.add_column([self.sampling], name='sampling')
        prop_table.add_column([self.stop], name='stop')
        prop_table.add_column([self.history], name='history')
        prop_table.add_column([self.sfh], name='sfh')
        prop_table.add_column([self.timescale], name='timescale')
        prop_table.add_column([self.efficiency], name='efficiency')
        prop_table.add_column([self.distance], name='distance')
        prop_table.add_column([self.wav], name='wav')
        prop_table.add_column([self.apertures], name='ap')
        prop_table.add_column([self.res_props[0]], name='tres')
        prop_table.add_column([self.res_props[1]], name='rres')
        prop_table.add_column([self.res_props[2]], name='mu')
        prop_table.add_column([self.n_members], name='n_members')
        prop_table.add_column([self.member_masses], name='member_masses')
        prop_table.add_column([self.inclinations], name='inclinations')
        prop_table.add_column([self.binaries], name='binaries')
        prop_table.add_column([self.end_times], name='end_times')
        prop_table.add_column([self.max_time], name='max_time')
        prop_table.add_column([self.offsets], name='offsets')

        with h5py.File(f'{filename}.hdf5', open_str) as out_file:
            write_table_hdf5(prop_table, out_file, path='properties',
                             compression=True, serialize_meta=True)
