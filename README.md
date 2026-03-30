![PyPI](https://img.shields.io/pypi/v/crflux)
![Tests](https://github.com/afedynitch/crflux/actions/workflows/tests.yml/badge.svg)

# `crflux.models`:  parameterizations of the Cosmic Ray flux

Historically, this module was part of the research code for a paper
[A. Fedynitch, J. Becker Tjus, and P. Desiati, Phys. Rev. D 86, 114024
(2012)](http://journals.aps.org/prd/abstract/10.1103/PhysRevD.86.114024),
where we compared the effects of different Cosmic Ray Flux models on the atmospheric lepton flux.

This code is now an integral dependency of the atmospheric neutrino and
air-shower cascade research code [MCEq: Matrix Cascade Equations (MCEq)](https://github.com/afedynitch/MCEq) providing:

- numerical models/parameterizations of high energy cosmic ray fluxes,
- conversions from all-particle into all-nucleon flux,
- geomagnetic cutoff support (rigidity cutoff in GV),
- and other convenience functions for semi-analytical atmospheric lepton flux calculations.

## Documentation

[Documentation](https://afedynitch.github.io/crflux). Please acknowledge the code by citing the paper above.

## Requirements

Python >= 3.9, numpy, scipy.

## Installation

    pip install crflux

## Example

```python
import numpy as np
import crflux.models as mods

model = mods.HillasGaisser2012("H3a")
E = np.logspace(1, 11, 100)
pfrac, p_flux, n_flux = model.p_and_n_flux(E)

# With geomagnetic cutoff of 7 GV
model_cut = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
pfrac_cut, p_cut, n_cut = model_cut.p_and_n_flux(E)
```

## Contributors

Hans Dembinski [@HDembinski](https://github.com/HDembinski)

## [MIT LICENSE](LICENSE)

Code and documentation copyright 2015 Anatoli Fedynitch
