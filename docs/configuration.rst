Configuration
=============

This page goes over the available configuration options for creating new protocluster models.

IMF
---

The first step to creating a new protocluster model (beyond picking a total mass budget) is sampling cluster members from an IMF. ``protopop`` makes use of the IMF modeling and sampling code `imf <https://github.com/keflavich/imf>`_ (`Richardson+ 2026 <{link}>`_) to do this sampling. To specify the mass function from which to draw members, ``protopop`` accepts either a string corresponding to one of ``imf``'s predefined functions (e.g. ``'salpeter'``) or a custom ``MassFunction``. For more details, see the `imf documentation <{link}>`_.

Cluster members are sampled from the provided IMF using the `imf <https://github.com/keflavich/imf>`_ software package. This package provides a great deal of flexibility for how sampling will performed, so additional keyword arguments can be supplied to new ``Cluster`` objects to guide the sampling. A brief overview of the options is given here; more detail can be found in the `imf documentation <{link}>`_.

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

text

* ``'is'``:
* ``'tc'``:
* ``'ca'``:
* ``'exp'``:

Tapered versions of each M10 history also exist, where a tapering factor of :math:`1-(t/t_f)` is applied to the base history. To access these, use ``'taper_is'``, ``'taper_tc'``, or ``'taper_ca'``.

Star formation history
----------------------

text

* ``'constant'``:
* ``'normalstart'``:
* ``'normalend'``:

Efficiency
----------

text
