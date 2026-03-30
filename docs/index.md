# crflux

Parameterizations of the cosmic ray flux at the top of Earth's atmosphere.

## Installation

```bash
pip install crflux
```

## Quick Start

```python
import numpy as np
import crflux.models as mods

# Create a flux model
model = mods.HillasGaisser2012("H3a")

# Evaluate nucleon flux over an energy range
E = np.logspace(1, 11, 100)  # GeV
pfrac, p_flux, n_flux = model.p_and_n_flux(E)

# With geomagnetic cutoff (rigidity cutoff in GV)
model_cut = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
pfrac_cut, p_cut, n_cut = model_cut.p_and_n_flux(E)
```

## Requirements

Python >= 3.9, numpy, scipy.

## Citation

If you use this code, please cite:
A. Fedynitch, J. Becker Tjus, and P. Desiati, Phys. Rev. D 86, 114024 (2012).

## License

[MIT License](https://github.com/afedynitch/crflux/blob/main/LICENSE)
