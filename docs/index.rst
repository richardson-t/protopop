.. protopop documentation master file, created by
   sphinx-quickstart on Wed Jun 17 10:28:04 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

protopop
========

Straightforward sampling of the evolutionary histories/expected fluxes of protocluster members. Documentation is under construction.

Installation
------------

To install ``protopop``: ::

  pip install protopop

To retrieve the data ``protopop`` uses to sample, run ``get-protopop-data`` from the command line within the environment where ``protopop`` is installed. Data consists of protostellar evolutionary tracks/corresponding flux tracks and a pre-made set of sample clusters. Tracks will download and unzip to a directory within ``protopop`` where the code expects to find the track information. Clusters will download and unzip to the current working directory, as this data is intended for direct access by users.

.. note::

   You can run ``get-protopop-tracks`` or ``get-protopop-clusters`` to retrieve the tracks or clusters individually.
   
Usage
-----

This section walks through a sample usage of ``protopop``. ``protopop``'s primary access point for users is through its ``Cluster`` class, which can be accessed with: ::

  from protopop.cluster import Cluster

``protopop`` provides an archive of pre-made clusters (see Installation) which can be easily read in and used. ::

  cluster_modeldir = /home/richardson-t/cluster_data
  cl = Cluster.read(f'{cluster_modeldir}/sample_cluster.hdf5')
  
``Clusters`` are composed of members with self-consistent evolutionary histories and accompanying flux tracks; these can be retrieved with ``Cluster.sample_ev`` and ``Cluster.sample_flux``, respectively. ``sample_ev`` requires a time at which to sample the cluster and will return a table with the evolutionary information about each member. Times should be in ``astropy`` time units. ::

  from astropy import units as u
  sample_time = 1 * u.Myr

  ev_table = cl.sample_ev(sample_time)

Sampling flux requires a time, a physical radius for a viewing aperture (or the radii of multiple apertures), and a wavelength/frequency (or set of wavelengths/frequencies) at which to sample. All of these properties should be in appropriate ``astropy`` units. ``sample_flux`` returns a float for a single aperture and sampling point, or an array of size (``n_apertures``, ``n_rows``). ::

  import numpy as np
  sample_aps = np.array([200, 2000]) * u.AU
  sample_wavs = np.array([4, 21, 1000]) * u.um

  fluxes = cl.sample_flux(sample_time, wav=sample_wavs, ap=sample_aps)
  
New clusters can be made by constructing a new ``Cluster`` object. A complete list of options for cluster construction and more details on the available attributes can be found in the API.

Details
-------

.. toctree::
   :maxdepth: 2

   configuration.rst
   api/api.rst
   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
