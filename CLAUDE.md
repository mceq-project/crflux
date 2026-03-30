# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`crflux` is a Python package providing parameterizations of the cosmic ray flux at the top of Earth's atmosphere. It is a dependency of [MCEq (Matrix Cascade Equations)](https://github.com/afedynitch/MCEq). The package provides numerical models of high-energy cosmic ray fluxes, all-particle to all-nucleon flux conversions, geomagnetic cutoff support, and convenience functions for atmospheric lepton flux calculations.

## Commands

- **Install**: `pip install -e ".[test]"`
- **Run tests**: `pytest tests/`
- **Run single test**: `pytest tests/test_models.py::TestNucleusFlux::test_scalar_input`
- **Coverage**: `pytest --cov=crflux --cov-report=term-missing`
- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Build docs**: `pip install -e ".[docs]" && mkdocs build`

## Architecture

The entire model library lives in a single module: `crflux/models.py`.

- **`PrimaryFlux`** (abstract base class): All flux models inherit from this. The abstract method is `_nucleus_flux(corsika_id, E)`. The public `nucleus_flux(corsika_id, E)` wraps it and applies optional geomagnetic cutoff. Shared methods: `total_flux(E)`, `tot_nucleon_flux(E)`, `p_and_n_flux(E)`, `lnA(E)`, `Z_A(corsika_id)`.
- **Nucleus IDs**: Follow CORSIKA scheme. Protons = 14, composite nuclei = `100*A + Z`.
- **All `nucleus_flux` implementations must handle array inputs** (use `np.atleast_1d`). No `np.vectorize`.
- **`GlobalSplineFitBeta`**: Uses bundled spline file (`crflux/GSF_spline_20171007.pkl.bz2`). Provides only nucleon fluxes via `p_and_n_flux`, not per-nucleus fluxes. Overrides base class methods directly.
- **Geomagnetic cutoff**: All models accept `geomagnetic_cutoff` (in GV) as a constructor kwarg. For a nucleus with charge Z, flux is zeroed below `E_cut = Z * R_cut` GeV.

## Dependencies

Runtime: numpy, scipy. Tests: pytest, pytest-cov, matplotlib. Docs: mkdocs-material, mkdocstrings.
