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
    for call in (model.nucleus_flux, model.total_flux, model.lnA):
        with pytest.raises(NotImplementedError, match="GlobalSplineFit"):
            call(*((14, energy) if call == model.nucleus_flux else (energy,)))


def test_reject_kinetic_energy_and_bad_theta():
    kinetic = gsf.ReducedGSF(gsf.GSFKineticEnergyPerNucleon(version="2026.1"))
    with pytest.raises(ValueError, match="total energy"):
        GlobalSplineFitReduced(reduction=kinetic)
    red = gsf.ReducedGSF()
    with pytest.raises(ValueError, match="finite components"):
        GlobalSplineFitReduced(reduction=red, theta=[0])


def test_beta_is_gsf2017_at_july_1998():
    """The frozen GSF2017 table uses the legacy convention: the atomic weight A sets
    the nucleus energy (A E_N) and the nucleon count (globalsplinefit < 2.1)."""
    from crflux.models import GlobalSplineFitBeta

    with pytest.warns(DeprecationWarning):
        beta = GlobalSplineFitBeta()
    model = gsf.GSFEnergy(version="2017")
    energy = np.logspace(1, 9, 25)
    proton = sum(
        model.flux(energy * model.z_to_a[s], s, time_interval=(199807, 199808))
        * model.z_to_a[s]
        * s[0]
        for g in model.active_groups
        for s in model._target_sids(model._resolve_z(g)[0])
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


@pytest.mark.parametrize("cutoff", [None, 7.0])
def test_full_model_legacy_iron_id(cutoff):
    from crflux.models import GlobalSplineFit

    model = GlobalSplineFit(version="2026.1", geomagnetic_cutoff=cutoff)
    energy = np.array([150.0, 1e3, 1e6])
    iron = model.nucleus_flux(5626, energy)
    assert np.all(iron[1:] > 0)
    np.testing.assert_array_equal(model.nucleus_flux(5426, energy), iron)


def test_zero_cutoff_is_no_cutoff():
    """0 GV must not trigger globalsplinefit's sigmoid (it cuts ~2 % of p)."""
    from crflux.models import GlobalSplineFit

    energy = np.logspace(0, 3, 30)
    ref = GlobalSplineFit(version="2026.1")
    zero = GlobalSplineFit(version="2026.1", geomagnetic_cutoff=0.0)
    for cid in (14, 402, 5626):
        np.testing.assert_array_equal(
            zero.nucleus_flux(cid, energy), ref.nucleus_flux(cid, energy)
        )
    np.testing.assert_array_equal(zero.p_and_n_flux(energy), ref.p_and_n_flux(energy))
    red0 = GlobalSplineFitReduced(version="2026.1", geomagnetic_cutoff=0.0)
    red = GlobalSplineFitReduced(version="2026.1")
    np.testing.assert_allclose(red0.p_and_n_flux(energy), red.p_and_n_flux(energy))


def test_cutoff_width():
    from crflux.models import NUCLEON_MASS_GEV, GlobalSplineFit

    rc = 8.0
    sharp = GlobalSplineFit(version="2026.1", geomagnetic_cutoff=rc, cutoff_width=0)
    smooth = GlobalSplineFit(version="2026.1", geomagnetic_cutoff=rc)
    ref = GlobalSplineFit(version="2026.1")
    for z, a, cid in [(1, 1, 14), (2, 4, 402), (26, 56, 5626)]:
        e_cut = np.sqrt((z * rc) ** 2 + (a * NUCLEON_MASS_GEV) ** 2)
        energy = e_cut * np.array([0.999, 1.001])
        below, above = sharp.nucleus_flux(cid, energy)
        assert below == 0.0
        np.testing.assert_allclose(above, ref.nucleus_flux(cid, energy)[1])
        # default 1 GV sigmoid: half of the flux at the cutoff rigidity
        np.testing.assert_allclose(
            smooth.nucleus_flux(cid, e_cut),
            0.5 * ref.nucleus_flux(cid, e_cut),
            rtol=2e-3,
        )
    red = GlobalSplineFitReduced(
        version="2026.1", geomagnetic_cutoff=rc, cutoff_width=0.5
    )
    assert red.gsf.cutoff_width == 0.5
    with pytest.raises(ValueError, match="supplied reduction"):
        GlobalSplineFitReduced(reduction=gsf.ReducedGSF(), cutoff_width=0.0)


def test_delta_0_from_nucleon_fluxes():
    from crflux.models import GlobalSplineFit

    energy = np.logspace(1, 9, 25)
    for model in (GlobalSplineFit(version="2026.1"), GlobalSplineFitReduced()):
        _, p, n = model.p_and_n_flux(energy)
        np.testing.assert_allclose(model.delta_0(energy), (p - n) / (p + n))
