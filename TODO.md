# TODO

## High Priority

- [x] Replace `GlobalSplineFit` with new GSF implementation: `GlobalSplineFit` + `GlobalSplineFitReduced` (via globalsplinefit)
- [ ] Regenerate `GSF_spline_*.pkl.bz2` with modern scipy to fix `DeprecationWarning` on unpickling (scipy.interpolate.fitpack2 namespace removal in SciPy 2.0)
- [x] Publish to PyPI (2.0.0, 2.1.0)

## Medium Priority

- [ ] Verify GitHub Pages serves correctly at mceq-project.github.io/crflux
- [ ] Add type hints throughout `crflux/models.py`
- [ ] Add uncertainty/covariance support for flux models (partial implementation existed in old `GlobalSplineFit`)
- [ ] Consider solar modulation parameter (related to geomagnetic cutoff feature)

## Low Priority

- [ ] Add coverage badge to README
- [ ] Add `py.typed` marker for PEP 561
- [ ] Rename `_BenzviMontaruli` to public API or remove if truly deprecated
- [ ] Consider splitting large `models.py` into separate modules per model family
