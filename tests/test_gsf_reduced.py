import numpy as np
import pytest

from crflux.models import GlobalSplineFitReduced

gsf = pytest.importorskip("globalsplinefit")


def test_reduced_nucleon_flux_and_sigma_variation():
    red = gsf.ReducedGSF(gsf.GSFEnergyPerNucleon(version="2026.1"))
    model = GlobalSplineFitReduced(reduction=red)
    energy = np.logspace(0, 8, 30)
    frac, p, n = model.p_and_n_flux(energy)
    np.testing.assert_allclose([p, n], red.flux(energy), rtol=1e-14)
    np.testing.assert_allclose(frac, p / (p + n))
    np.testing.assert_allclose(model.tot_nucleon_flux(energy), p + n)
    theta = np.zeros(red.n_params)
    theta[3] = 0.05 * red.sigma[3]
    varied = GlobalSplineFitReduced(reduction=red, theta=theta)
    np.testing.assert_allclose(varied.p_and_n_flux(energy)[1:], red.flux(energy, theta))
    theta[3] = 0  # constructor owns its vector
    assert varied.theta[3] != 0
    assert model.p_and_n_flux(10.0)[0].shape == (1,)
    with pytest.raises(NotImplementedError):
        model.nucleus_flux(14, energy)


def test_reject_kinetic_energy_and_bad_theta():
    kinetic = gsf.ReducedGSF(gsf.GSFKineticEnergyPerNucleon(version="2026.1"))
    with pytest.raises(ValueError, match="total energy"):
        GlobalSplineFitReduced(reduction=kinetic)
    red = gsf.ReducedGSF()
    with pytest.raises(ValueError, match="finite components"):
        GlobalSplineFitReduced(reduction=red, theta=[0])


def test_beta_is_gsf2017_at_july_1998():
    from importlib.metadata import version

    if tuple(int(x) for x in version("globalsplinefit").split(".")[:3]) < (2, 0, 1):
        pytest.skip("globalsplinefit < 2.0.1 ships a different '2017' set")
    from crflux.models import GlobalSplineFitBeta

    with pytest.warns(DeprecationWarning):
        beta = GlobalSplineFitBeta()
    model = gsf.GSFEnergyPerNucleon(version="2017")
    energy = np.logspace(1, 9, 25)
    proton = sum(
        model.p_and_n_flux(energy, g, time_interval=(199807, 199808))[0]
        for g in model.active_groups
    )
    np.testing.assert_allclose(beta.p_and_n_flux(energy)[1], proton, rtol=5e-3)


def test_full_model_matches_globalsplinefit():
    from crflux.models import GlobalSplineFit

    model = GlobalSplineFit(version="2026.1")
    ref = gsf.GSFEnergy(version="2026.1")
    energy = np.logspace(1, 10, 40)
    assert {14, 201, 402, 1608, 5626}.issubset(model.nucleus_ids)
    np.testing.assert_allclose(model.nucleus_flux(14, energy), ref.flux(energy, "p"))
    np.testing.assert_allclose(model.nucleus_flux(5626, energy), ref.flux(energy, "Fe"))
    np.testing.assert_allclose(
        model.total_flux(energy), ref.total_flux(energy), rtol=1e-12
    )
    assert np.all(model.nucleus_flux(999, energy) == 0)
    nucleon = gsf.GSFEnergyPerNucleon(version="2026.1")
    p, n = sum(nucleon.p_and_n_flux(energy, g) for g in nucleon.active_groups)
    frac, mp, mn = model.p_and_n_flux(energy)
    np.testing.assert_allclose([mp, mn], [p, n])
    np.testing.assert_allclose(model.tot_nucleon_flux(energy), p + n)
    np.testing.assert_allclose(frac, p / (p + n))
    # reduced model at theta = 0 is the full nucleon flux
    reduced = GlobalSplineFitReduced(version="2026.1")
    np.testing.assert_allclose(reduced.p_and_n_flux(energy)[1:], [p, n], rtol=1e-10)
    assert reduced.nucleus_ids == []
    # MCEq checks the direct bases by name
    for obj in (model, reduced):
        assert any(b.__name__ == "PrimaryFlux" for b in type(obj).__bases__)


def test_full_model_versions_and_cutoff():
    from crflux.models import GlobalSplineFit

    old = GlobalSplineFit(version="2017")
    assert 201 not in old.nucleus_ids and old.sname == "GSF2017"
    cut = GlobalSplineFit(version="2026.1", geomagnetic_cutoff=10.0)
    energy = np.array([2.0, 5.0, 1e3])
    assert cut.geomagnetic_cutoff == 10.0
    assert cut.nucleus_flux(14, energy)[0] < 1e-3 * old.nucleus_flux(14, energy)[0]
    assert cut.nucleus_flux(14, energy)[2] > 0
    with pytest.raises(ValueError, match="supplied reduction"):
        GlobalSplineFitReduced(reduction=gsf.ReducedGSF(), geomagnetic_cutoff=5.0)
