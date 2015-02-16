.. CRFluxModels documentation master file, created by
   sphinx-quickstart on Sun Nov 30 18:10:29 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the :mod:`CRFluxModels`
=========================================

Historically, this module was part of the research code for a paper 
(A. Fedynitch, J. Becker Tjus, and P. Desiati, Phys. Rev. D 86, 114024 (2012)), 
where we compared the effects of different primary Cosmic Ray Flux models - therefore
the name. With time the module became an integral part of my research and has been
recently cleaned up and documented. It does the following things:

- implements numerical models of high energy cosmic ray fluxes, 
- converts from all-particle into all-nucleon flux,
- includes some convenience functions for semi-analytical atmospheric lepton flux calculations.

I never encountered situations in my work where this module was the performance
bottle-neck. The formulas are therefore written in a human-readable way and were not 
optimized for speed. Due to historical reasons some methods might be also not very elegantly written.


.. toctree::
   :maxdepth: 2

````````````````````
Module documentation
````````````````````

.. automodule:: CRFluxModels
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

