Configuration
=============

Some parameters of ``protopop``'s protocluster models adjust the theoretical assumptions shaping cluster construction. These "theory parameters" place some limitations on the values they will accept; this page goes over the available configuration options for each when creating new protoclusters.

IMF
---

The first step to creating a new protocluster model (beyond picking a total mass budget) is sampling cluster members from an IMF. ``protopop`` makes use of the IMF modeling and sampling code `imf <https://github.com/keflavich/imf>`_ (`Richardson+ 2026 <{link}>`_) to do this sampling. To specify the mass function from which to draw members, ``protopop`` provides an ``'imf'`` keyword which accepts either a string corresponding to one of ``imf``'s predefined functions (e.g. ``'salpeter'``) or a custom ``MassFunction``. For more details, see the `imf documentation <{https://imf.readthedocs.io/en/latest}>`_.

Cluster members are sampled from the provided IMF using the `imf <https://github.com/keflavich/imf>`_ software package. This package provides a great deal of flexibility for how sampling will performed, so additional keyword arguments can be supplied to new ``Cluster`` objects to guide the sampling. A brief overview of the options is given here; more detail can be found in the `imf documentation <https://imf.readthedocs.io/en/latest}>`_.

* Sampling method (``sampling`` keyword)

  * ``'random'``: Random sampling.
  * ``'optimal'``: Optimal sampling; a method which produces a stellar population perfectly following the underlying distribution. (See `Kroupa+ 2013 <https://doi.org/10.48550/arXiv.1112.3340>`_.)

* Stop criterion (``stop_criterion`` keyword). This only matters for random sampling, as the stop condition for random sampling is reaching the mass budget provided to the ``Cluster``; this determines what to do with any stars exceeding the mass budget.

  * ``'nearest'``: Keep all sampled stars (in order sampled) whose total mass is closest to the final mass budget.
  * ``'before'``: Keep all sampled stars (in order sampled) whose total mass is under the final mass budget. 
  * ``'after'``: Keep all sampled stars (in order sampled) whose total mass is under the final mass budget, plus the next one.
  * ``'sorted'``: Sort all sampled stars in increasing mass order, then use ``'nearest'.``

Accretion history
-----------------

Each cluster member is assumed to follow the same "accretion history", i.e. gains mass following the same theory of protostellar mass accretion. ``protopop`` provides a number of accretion history models derived from `McKee & Offner (2010, M10) <https://doi.org/10.1088/0004-637X/716/1/167>`_ and `Duarte-Cabral+ (2013, D13) <https://doi.org/10.1051/0004-6361/201321393>`_

The following histories are available and can be provided as arguments to the ``'history'`` keyword:

* ``'is'``: Isothermal sphere (i.e. constant rate) accretion. [M10]_
* ``'tc'``: Turbulent core accretion: accelerating accretion dependent on the current and final stellar mass. [M10]_
* ``'ca'``: Parameterization of competitive accretion; all stars form in a roughly constant time. Like turbulent core, accelerates based on the current and final stellar mass. [M10]_
* ``'exp'``: Exponentially tapered accretion. [D13]_

Tapered versions of each M10 history also exist, where a tapering factor of :math:`1-(t/t_f)` is applied to the base history. To access these, use ``'taper_is'``, ``'taper_tc'``, or ``'taper_ca'``.

Star formation history
----------------------

Part of cluster creation is selecting a star formation history (SFH), which determines when cluster members begin accreting relative to each other. SFHs are implemented as probability distributions from which times are drawn; these times set the beginning of formation :math:`t=0` and therefore the age of each cluster member. (Without the assumption of an SFH, each cluster member would begin forming at the same time.)

Entering an SFH for a new cluster requires specifying a type (``'sfh'``) and timescale (``'timescale'``), where the timescale is the characteristic time for star formation :math:`t_{\rm sf}` controlling the shape of the probability distribution. ``protopop`` provides the following options for SFH types:

* ``'constant'``: Uniform distribution with a max time of :math:`t_{\rm sf}`.
* ``'normalstart'``: Normal distribution with a :math:`1 - \sigma` width of :math:`t_{\rm sf}`.
* ``'normalend'``: Same as ``'normalstart'``, but all cluster members are first aligned such that all formation ends at the same time. (This matters in cases where the accretion time is mass-dependent, which is true for e.g. isothermal-sphere accretion.)

Efficiency
----------

Mass accretion efficiency (:math:`\epsilon_{\rm sf}`) determines how much mass is modeled as being transferred from a prestellar mass reservoir to an eventual star. This value is reflected in the predicted fluxes in the tracks ``protopop`` uses to model the flux evolution of cluster members and therefore has an impact on the resulting predictions. ``protopop`` permits efficiencies of 33 and 100, which are treated as percentage values (i.e. an efficiency of 33 means that 33% of initial mass will be accreted); this value is held constant for an entire cluster regardless of eventual stellar mass.

.. [M10] `McKee & Offner (2010) <https://doi.org/10.1088/0004-637X/716/1/167>`_
.. [D13] `Duarte-Cabral+ (2013) <https://doi.org/10.1051/0004-6361/201321393>`_
