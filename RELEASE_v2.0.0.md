# crflux v2.0.0

Major modernization release.

## New Features

- **Geomagnetic cutoff support**: All flux models now accept a `geomagnetic_cutoff` parameter (rigidity cutoff in GV). For a nucleus with charge Z, the flux is zeroed below E_cut = Z * R_cut.
  ```python
  model = HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
  ```

## Bug Fixes

- **Fix NumPy 2.x compatibility**: Removed `np.vectorize` from all base class methods. All `nucleus_flux` implementations now handle array inputs natively, fixing `ValueError: setting an array element with a sequence` on NumPy >= 2.0.
- **Fix integer division in `Z_A`**: Mass number A is now correctly returned as an integer.
- **Fix bare `except:` clauses** in `ZatsepinSokolskaya` and `_BenzviMontaruli` with proper array handling via `np.atleast_1d`.
- **Fix `_find_nearby_id`**: Now respects the `delta_A` parameter instead of using a hardcoded value.

## Breaking Changes

- **Python >= 3.9 required** (Python 2 support and `six` dependency removed).
- **`nucleus_flux()` now always returns `np.ndarray`**, even for scalar input.
- **`GlobalSplineFit` class removed** (depended on the unpackaged `gsf` module). Use `GlobalSplineFitBeta` instead.
- Subclasses now implement `_nucleus_flux()` instead of `nucleus_flux()`. The public `nucleus_flux()` method on `PrimaryFlux` is a concrete wrapper that applies the geomagnetic cutoff. Direct callers of `model.nucleus_flux(...)` are unaffected.

## Infrastructure

- Migrated from `setup.py` to `pyproject.toml` (PEP 621).
- Migrated CI from Azure Pipelines to GitHub Actions (Python 3.9-3.13).
- Added `ruff` for linting and formatting with CI conformance checks.
- Migrated documentation from Sphinx/RST to MkDocs Material with mkdocstrings, auto-deployed to GitHub Pages.
- Test suite: 274 tests at 94% coverage (previously 1 broken test).
- Default branch renamed from `master` to `main`.
