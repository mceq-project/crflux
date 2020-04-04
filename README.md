![PyPI](https://img.shields.io/pypi/v/crflux)
[![Documentation](https://readthedocs.org/projects/crfluxmodels/badge/?version=latest)](https://crfluxmodels.readthedocs.io/en/latest/?badge=latest)
![Azure DevOps builds](https://img.shields.io/azure-devops/build/afedynitch/MCEq/7)
![Azure DevOps tests](https://img.shields.io/azure-devops/tests/afedynitch/MCEq/7)

# `crflux.models`:  parameterizations of the Cosmic Ray flux

Historically, this module was part of the research code for a paper 
[A. Fedynitch, J. Becker Tjus, and P. Desiati, Phys. Rev. D 86, 114024 
(2012)](http://journals.aps.org/prd/abstract/10.1103/PhysRevD.86.114024), 
where we compared the effects of different Cosmic Ray Flux models on the atmospheric lepton flux.

This code is now an integral dependency of the atmospheric neutrino and
air-shower cascade research code [MCEq: Matrix Cascade Equations (MCEq)](https://github.com/afedynitch/MCEq) providing: 

- numerical models/parameterizations of high energy cosmic ray fluxes, 
- conversions from all-particle into all-nucleon flux,
- and other convenience functions for semi-analytical atmospheric lepton flux calculations.

Performance was never a bottle-neck, thus it's not the most numerically brilliant code to date but it does the job.

If you find it too slow for your applications feel free to open an issue or send a pull request or just branch/fork and let me know.

## Documentation

[The documentation is located here](http://crfluxmodels.readthedocs.org/en/latest/index.html). Please acknowledge the code by citing the paper above.

## Requirements


Python 2 or 3, numpy, scipy, matplotlib.

## Installation

    pip install crflux

## Example

The interaction with the classes and plotting is demonstrated in the `test()` method::

    from crflux.models import *
    test()

## Contributors

Hans Dembinski [@HDembinski](https://github.com/HDembinski)

## [MIT LICENSE](LICENSE)


Code and documentation copyright 2015 Anatoli Fedynitch

