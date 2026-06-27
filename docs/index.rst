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

To retrieve the data ``protopop`` uses to sample, run ``get-protopop-data`` from the command line within the environment where ``protopop`` is installed.

.. note::

   Data consists of protostellar evolutionary tracks/corresponding flux tracks and a pre-made set of sample clusters; you can run ``get-protopop-tracks`` or ``get-protopop-clusters`` to retrieve them individually.

Usage
-----

``protopop``'s primary access point for users is through its ``Cluster`` class, which can be accessed with: ::

  from protopop.cluster import Cluster

To sample with ``protopop``...

To make a new cluster...

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
