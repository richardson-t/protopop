Configuration
=============

This page goes over the available configuration options for creating new protocluster models.

IMF
---

text

This is

* Sampling method

  * ``'random'``:
  * ``'optimal'``:

* Stop criterion (for random sampling only)

  * ``'nearest'``:
  * ``'before'``:
  * ``'after'``:
  * ``'sorted'``:

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
