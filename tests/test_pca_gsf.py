"""
Test suite for PCAGlobalSplineFit model.

These tests validate that the PCA-based GSF model works correctly,
including component variations and flux calculations.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


# Only run these tests if we can import the model
# (may fail if pickle file doesn't exist yet)
try:
    from crflux.models import PCAGlobalSplineFit
    PCAGSF_AVAILABLE = True
except (ImportError, FileNotFoundError):
    PCAGSF_AVAILABLE = False


@pytest.mark.skipif(not PCAGSF_AVAILABLE, reason="PCA_GSF9_2025.pkl file not found")
class TestPCAGlobalSplineFit:
    """Test PCAGlobalSplineFit model."""

    @pytest.fixture
    def energies(self):
        """Common energy grid for testing."""
        return np.logspace(2, 6, 10)  # 100 GeV to 1 PeV

    @pytest.fixture
    def model(self):
        """Default PCA GSF model instance."""
        return PCAGlobalSplineFit()

    def test_initialization(self, model):
        """Test that model initializes correctly."""
        assert hasattr(model, 'x')
        assert hasattr(model, 'L')  # Low-rank factor
        assert hasattr(model, 'D')  # Diagonal component
        assert hasattr(model, 'phi')  # Latent parameters
        assert hasattr(model, 'phi_mean')
        assert hasattr(model, 'phi_std')
        assert hasattr(model, 'phi_cov')  # Latent parameter covariance
        assert hasattr(model, 'central_flux')
        assert hasattr(model, 'n_components')
        assert model.n_components > 0

    def test_p_and_n_flux_returns_tuple(self, model, energies):
        """Test that p_and_n_flux returns a tuple of three elements."""
        result = model.p_and_n_flux(energies)
        assert isinstance(result, tuple)
        assert len(result) == 3
        p_frac, p_flux, n_flux = result
        assert p_frac.shape == energies.shape
        assert p_flux.shape == energies.shape
        assert n_flux.shape == energies.shape

    def test_p_and_n_flux_scalar_input(self, model):
        """Test that p_and_n_flux works with scalar input."""
        E = 1000.0  # 1 TeV
        result = model.p_and_n_flux(E)
        assert isinstance(result, tuple)
        assert len(result) == 3
        p_frac, p_flux, n_flux = result
        # Should return scalars for scalar input
        assert np.isscalar(p_frac)
        assert np.isscalar(p_flux)
        assert np.isscalar(n_flux)

    def test_proton_fraction_bounds(self, model, energies):
        """Test that proton fraction is between 0 and 1."""
        p_frac, _, _ = model.p_and_n_flux(energies)
        assert np.all(p_frac >= 0)
        assert np.all(p_frac <= 1)

    def test_flux_positivity(self, model, energies):
        """Test that fluxes are non-negative."""
        _, p_flux, n_flux = model.p_and_n_flux(energies)
        assert np.all(p_flux >= 0)
        assert np.all(n_flux >= 0)

    def test_tot_nucleon_flux(self, model, energies):
        """Test that total nucleon flux equals sum of p and n fluxes."""
        _, p_flux, n_flux = model.p_and_n_flux(energies)
        tot_flux = model.tot_nucleon_flux(energies)
        assert_allclose(tot_flux, p_flux + n_flux, rtol=1e-10)

    def test_nucleus_flux_returns_zeros(self, model, energies):
        """Test that nucleus_flux returns zeros (not supported)."""
        flux = model.nucleus_flux(14, energies)  # Proton CORSIKA ID
        assert_array_equal(flux, np.zeros_like(energies))

    def test_var_component_changes_flux(self, model):
        """Test that varying a component changes the flux."""
        E = 1000.0
        _, p_flux_orig, _ = model.p_and_n_flux(E)
        
        # Apply variation to first component
        model.var_component(1, 1.0)
        _, p_flux_var, _ = model.p_and_n_flux(E)
        
        # Flux should change
        assert p_flux_var != p_flux_orig

    def test_reset_components(self, model):
        """Test that reset_components returns to original flux."""
        E = 1000.0
        _, p_flux_orig, _ = model.p_and_n_flux(E)
        
        # Apply variation
        model.var_component(1, 1.0)
        _, p_flux_var, _ = model.p_and_n_flux(E)
        assert p_flux_var != p_flux_orig
        
        # Reset and check we're back to original
        model.reset_components()
        _, p_flux_reset, _ = model.p_and_n_flux(E)
        assert_allclose(p_flux_reset, p_flux_orig, rtol=1e-10)

    def test_var_component_bounds(self, model):
        """Test that var_component validates component number."""
        with pytest.raises(AssertionError):
            model.var_component(0, 1.0)  # Component 0 doesn't exist
        
        with pytest.raises(AssertionError):
            model.var_component(model.n_components + 1, 1.0)  # Too large

    def test_initialization_with_variation(self):
        """Test initialization with immediate variation."""
        model = PCAGlobalSplineFit(comp=1, delta=1.0)
        
        # Should have variation applied (phi should differ from phi_original)
        assert not np.allclose(model.phi, model.phi_original)
        
        # Should be able to reset
        model.reset_components()
        assert_allclose(model.phi, model.phi_original)

    def test_multiple_variations_additive(self, model):
        """Test that multiple variations are applied additively."""
        E = 1000.0
        
        # Get original
        _, p_flux_orig, _ = model.p_and_n_flux(E)
        
        # Apply first variation
        model.var_component(1, 1.0)
        _, p_flux_1, _ = model.p_and_n_flux(E)
        
        # Apply second variation (should be additive)
        model.var_component(1, 1.0)
        _, p_flux_2, _ = model.p_and_n_flux(E)
        
        # Second should be different from first
        assert p_flux_2 != p_flux_1
        assert p_flux_2 != p_flux_orig

    def test_flux_decreases_with_energy(self, model):
        """Test that flux generally decreases with energy (power-law behavior)."""
        energies = np.logspace(2, 6, 20)
        tot_flux = model.tot_nucleon_flux(energies)
        
        # Flux should generally decrease (check most consecutive pairs)
        decreasing_pairs = np.sum(np.diff(tot_flux) < 0)
        assert decreasing_pairs > len(energies) * 0.8  # At least 80% decreasing

    def test_name_attributes(self, model):
        """Test that model has proper name attributes."""
        assert hasattr(model, 'name')
        assert hasattr(model, 'sname')
        assert isinstance(model.name, str)
        assert isinstance(model.sname, str)

    def test_nucleus_ids_empty(self, model):
        """Test that nucleus_ids is empty (nucleon-only model)."""
        assert hasattr(model, 'nucleus_ids')
        assert len(model.nucleus_ids) == 0
