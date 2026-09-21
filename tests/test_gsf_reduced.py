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
