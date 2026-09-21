# Available Models

All models inherit from `PrimaryFlux` and implement the same interface.
They accept an optional `geomagnetic_cutoff` parameter (in GV) that zeros
the flux below the corresponding rigidity for each nucleus.

## Model Comparison

### Nucleon Flux

All-nucleon flux (proton + neutron) scaled by $E^{2.5}$ for visibility across the full energy range.

![Nucleon flux comparison](img/nucleon_flux.png)

### All-Particle Flux

Total flux of all nuclei (all-particle spectrum) scaled by $E^{2.5}$.

![Total particle flux comparison](img/total_flux.png)

### Neutron Fraction

Fraction of neutrons in the nucleon flux, showing composition differences between models.

![Neutron fraction](img/neutron_fraction.png)

### Mean Logarithmic Mass

$\langle\ln A\rangle$ as a function of energy per particle. Higher values indicate heavier composition.

![Mean log mass](img/lnA.png)

## Parameterized Models

| Class | Reference | Energy Range | Notes |
|-------|-----------|-------------|-------|
| `HillasGaisser2012` | Gaisser, Astropart. Phys. 35, 801 (2012) | Full range | Variants: `H3a`, `H4a` |
| `H3a_polygonato` | Modified Gaisser (2012) | Full range | Poly-gonato at low E |
| `GaisserStanevTilav` | Gaisser, Stanev, Tilav, arXiv:1303.3565 (2013) | Full range | Variants: `3-gen`, `4-gen` |
| `CombinedGHandHG` | Fedynitch et al., PRD 86, 114024 (2012) | Full range | GH at low E, HG at high E |
| `PolyGonato` | Hoerandel, Astropart. Phys. 19, 193 (2003) | Full range | 11 mass groups |
| `ZatsepinSokolskaya` | Zatsepin & Sokolskaya, A&A 458, 1 (2006) | < 1-10 PeV | Variants: `default`, `pamela` |
| `GaisserHonda` | Gaisser & Honda, ARNPS 52, 153 (2002) | < 100 TeV | Tuned to balloon data |
| `Thunman` | Thunman et al., Astropart. Phys. 5, 309 (1996) | Full range | Proton-only broken power law |
| `SimplePowerlaw27` | Based on Thunman | Below knee | Proton-only $E^{-2.7}$ |
| `GlobalSplineFitBeta` | Dembinski et al., PoS ICRC2017 533 | 10 GV - $10^{11}$ GeV | Spline-based, nucleon flux only |

## Nucleus ID Scheme (CORSIKA)

Protons have ID 14. Composite nuclei: `ID = 100 * A + Z`

| Nucleus | A | Z | CORSIKA ID |
|---------|---|---|-----------|
| Proton | 1 | 1 | 14 |
| Helium | 4 | 2 | 402 |
| Carbon | 12 | 6 | 1206 |
| Silicon | 28 | 14 | 2814 |
| Iron | 54 | 26 | 5426 |

## Geomagnetic Cutoff

All models accept `geomagnetic_cutoff` (in GV) as a constructor parameter.
For a nucleus with charge $Z$, the energy cutoff is $E_\text{cut} = Z \times R_\text{cut}$ (GeV).
Flux is zeroed below this energy.

```python
import crflux.models as mods

# 7 GV cutoff (typical for mid-latitudes)
model = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
```

The plot below shows the effect of different cutoff values on the H3a nucleon flux:

![Geomagnetic cutoff effect](img/geomagnetic_cutoff.png)

## GSF 2026 reduced nucleon model

`GlobalSplineFitReduced` adapts the optional `globalsplinefit` package's
energy-pivot proton/neutron model to the `PrimaryFlux` / MCEq interface.
Install the `gsf` extra, or install the maintained globalsplinefit checkout.

```python
from crflux.models import GlobalSplineFitReduced
import numpy as np

central = GlobalSplineFitReduced(version="2026.1")
red = central.reduction
# Parameters are relative-flux deviations; retain their full covariance.
theta = np.zeros(red.n_params)
theta[3] = 0.05 * red.sigma[3]
varied = GlobalSplineFitReduced(reduction=red, theta=theta)
fraction, proton, neutron = varied.p_and_n_flux(np.logspace(0, 8, 50))
```

The default uses the published 24-parameter p/n pivot grid and Solar Cycle 24
modulation average. Energies are total GeV per nucleon; the adapter does not
rescale the package's flux units. For a different time interval or a rigidity
cutoff, construct the GSF reduction with those settings first. A reduced model
has no individual-nucleus or composition prediction. Kinetic-energy reductions
are rejected to prevent a silent energy-convention mismatch.

The development GSF 2026.1 dependency currently requires maintainer access to
`gsf-project/globalsplinefit`. Run `pytest tests/test_gsf_reduced.py -q` in an
environment with that dependency installed. Public CI skips these optional
integration tests until the GSF package is publicly installable.
