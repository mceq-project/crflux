`crflux.models` --- models of the high-energy cosmic ray flux
-------------------------------------------------------------

Historically, this module was part of the research code for a paper 
`A. Fedynitch, J. Becker Tjus, and P. Desiati, Phys. Rev. D 86, 114024 
(2012) <http://journals.aps.org/prd/abstract/10.1103/PhysRevD.86.114024>`_, 
where we compared the effects of different Cosmic Ray Flux models - therefore
the name.

This code is now an integral dependency of the atmospheric neutrino and
air-shower cascade research code MCEq `Matrix Cascade Equation (MCEq) <https://github.com/afedynitch/MCEq>`_, providing: 

- numerical models/parameterizations of high energy cosmic ray fluxes, 
- conversions from all-particle into all-nucleon flux,
- other convenience functions for semi-analytical atmospheric lepton flux calculations.

Performance was never a bottle-neck, thus it's not the most numerically brilliant code to date. But it does the job!

If you find it too slow for your applications feel free to open an issue or send a pull request or just branch/fork and let me know.

Documentation and Citations
---------------------------

Checkout the `documentation <http://crfluxmodels.readthedocs.org/en/latest/index.html#>`_ for more in-depth info.
If you are using one of these models in your research, cite the paper referenced in the documentation. Feel free to acknowledge or footnote `crflux.models`.

Requirements
------------

Python 2 or 3, numpy, scipy, matplotlib

Installation
------------

pip install crflux

Example
-------

The interaction with the classes and plotting is demonstrated in the `test()` method::

    from crflux.models import *
    test()

Contributors
------------
Hans Dembinski (github:HDembinski)

MIT LICENSE
-----------

Code and documentation copyright 2014 Anatoli Fedynitch

