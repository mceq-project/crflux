"""
Test suite for Low-Rank Plus Diagonal error calculations and comparison with GlobalSplineFit2025.

These tests validate that the Low-Rank Plus Diagonal error calculation works correctly.
The Low-Rank Plus Diagonal approach (Σ ≈ L*L^T + D) exactly preserves marginal variances
through the diagonal term D, so errors should be very close to the full GSF model.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# Import models - some may not be available
try:
    from crflux.models import PCAGlobalSplineFit
    PCAGSF_AVAILABLE = True
except (ImportError, FileNotFoundError):
    PCAGSF_AVAILABLE = False

try:
    from crflux.models import GlobalSplineFit2025
    GSF2025_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GSF2025_AVAILABLE = False


@pytest.mark.skipif(not PCAGSF_AVAILABLE, reason="PCA GSF model not available")
class TestPCAErrors:
    """Test Low-Rank Plus Diagonal error calculation methods."""

    @pytest.fixture
    def energies(self):
        """Common energy grid for testing."""
        return np.logspace(2, 6, 10)  # 100 GeV to 1 PeV

    @pytest.fixture
    def model(self):
        """Default PCA GSF model instance."""
        return PCAGlobalSplineFit()

    def test_p_and_n_flux_error_returns_tuple(self, model, energies):
        """Test that p_and_n_flux_error returns a tuple of two elements."""
        result = model.p_and_n_flux_error(energies)
        assert isinstance(result, tuple)
        assert len(result) == 2
        p_err, n_err = result
        assert p_err.shape == energies.shape
        assert n_err.shape == energies.shape

    def test_p_and_n_flux_error_scalar_input(self, model):
        """Test that p_and_n_flux_error works with scalar input."""
        E = 1000.0  # 1 TeV
        result = model.p_and_n_flux_error(E)
        assert isinstance(result, tuple)
        assert len(result) == 2
        p_err, n_err = result
        # Should return scalars for scalar input
        assert np.isscalar(p_err)
        assert np.isscalar(n_err)

    def test_errors_positive(self, model, energies):
        """Test that all errors are non-negative."""
        p_err, n_err = model.p_and_n_flux_error(energies)
        assert np.all(p_err >= 0)
        assert np.all(n_err >= 0)

    def test_tot_nucleon_flux_error_positive(self, model, energies):
        """Test that total nucleon flux error is non-negative."""
        tot_err = model.tot_nucleon_flux_error(energies)
        assert np.all(tot_err >= 0)

    def test_tot_nucleon_flux_error_scalar(self, model):
        """Test that tot_nucleon_flux_error works with scalar input."""
        E = 1000.0
        tot_err = model.tot_nucleon_flux_error(E)
        assert np.isscalar(tot_err)
        assert tot_err >= 0

    def test_errors_smaller_than_flux(self, model, energies):
        """Test that errors are reasonable compared to flux values.
        
        Errors should typically be much smaller than the flux itself
        for a well-constrained model.
        """
        _, p_flux, n_flux = model.p_and_n_flux(energies)
        p_err, n_err = model.p_and_n_flux_error(energies)
        
        # Check that relative errors are reasonable (< 100%)
        # Avoid division by zero
        mask_p = p_flux > 0
        mask_n = n_flux > 0
        
        if np.any(mask_p):
            rel_err_p = p_err[mask_p] / p_flux[mask_p]
            assert np.all(rel_err_p < 1.0), "Proton flux relative error > 100%"
        
        if np.any(mask_n):
            rel_err_n = n_err[mask_n] / n_flux[mask_n]
            assert np.all(rel_err_n < 1.0), "Neutron flux relative error > 100%"

    def test_total_error_consistency(self, model, energies):
        """Test relationship between individual and total errors.
        
        Without correlations: Var(p+n) = Var(p) + Var(n)
        With positive correlation: Var(p+n) > Var(p) + Var(n)
        With negative correlation: Var(p+n) < Var(p) + Var(n)
        
        The total error should be in a reasonable range relative to
        the individual errors.
        """
        p_err, n_err = model.p_and_n_flux_error(energies)
        tot_err = model.tot_nucleon_flux_error(energies)
        
        # Total variance should be positive
        tot_var = tot_err**2
        assert np.all(tot_var >= 0)
        
        # In most cases, with positive correlations between p and n:
        # tot_err >= sqrt(p_err^2 + n_err^2)
        # But can be less with negative correlations
        
        # Check that total error is not unreasonably large
        # (should not be more than sqrt(2) * max(individual errors) in typical cases)
        max_individual = np.maximum(p_err, n_err)
        assert np.all(tot_err <= 3.0 * max_individual), \
            "Total error is unreasonably large compared to individual errors"

    def test_error_variation_with_component(self, model):
        """Test that varying a component doesn't significantly affect the error calculation.
        
        In the Low-Rank Plus Diagonal model, the diagonal term D preserves exact
        marginal variances regardless of the latent parameter values. However,
        the low-rank component contributes variance that depends on the Jacobian,
        which changes slightly when flux changes. The total error should remain
        very close to constant due to D dominance.
        """
        E = 1000.0
        
        # Get error at default
        err_default = model.tot_nucleon_flux_error(E)
        
        # Apply variation and check error again
        model.var_component(1, 1.0)
        err_varied = model.tot_nucleon_flux_error(E)
        
        # Errors should be very similar (within 2% due to interpolation and Jacobian effects)
        assert_allclose(err_varied, err_default, rtol=0.02)
        
        model.reset_components()


@pytest.mark.skipif(
    not (PCAGSF_AVAILABLE and GSF2025_AVAILABLE),
    reason="Both PCA GSF and GlobalSplineFit2025 required"
)
class TestPCAErrorsVsGSF2025:
    """Compare Low-Rank Plus Diagonal errors with GlobalSplineFit2025 errors.
    
    The Low-Rank Plus Diagonal approach exactly preserves marginal variances
    through the diagonal term D, so errors should be very close to the full
    GSF model (typically within 5% or better).
    """

    @pytest.fixture
    def energies(self):
        """Common energy grid for testing."""
        # Use a range where both models are well-defined
        return np.logspace(2, 6, 20)  # 100 GeV to 1 PeV

    @pytest.fixture
    def pca_model(self):
        """PCA GSF model instance."""
        return PCAGlobalSplineFit()

    @pytest.fixture
    def gsf_model(self):
        """GlobalSplineFit2025 model instance."""
        return GlobalSplineFit2025()

    def test_proton_and_neutron_error_comparison(self, pca_model, gsf_model, energies):
        """Compare proton and neutron flux errors between Low-Rank Plus Diagonal and GSF2025.
        
        The Low-Rank Plus Diagonal approach exactly preserves the diagonal of the
        covariance matrix through the D term, so marginal uncertainties should be
        within a few percent (95-105%) of the full GSF model.
        """
        # Get Low-Rank Plus Diagonal errors
        p_err_pca, n_err_pca = pca_model.p_and_n_flux_error(energies)
        
        # Get GSF2025 total nucleon errors summed over all groups
        from globalsplinefit import GSFEnergyPerNucleon
        gsf_nucleon = GSFEnergyPerNucleon(version="2025", use_approximate_solar_cycle_average=True)
        
        # Get total proton and neutron flux from GSF
        pn_flux_gsf = gsf_nucleon.p_and_n_total_flux(energies)
        p_flux_gsf = pn_flux_gsf[0]
        n_flux_gsf = pn_flux_gsf[1]
        
        # Calculate total errors by summing over all groups with covariance
        # Initialize error arrays
        n_energies = len(energies)
        cov_p_total = np.zeros((n_energies, n_energies))
        cov_n_total = np.zeros((n_energies, n_energies))
        
        # Sum covariances from all group pairs
        for group1 in gsf_nucleon.active_groups:
            for group2 in gsf_nucleon.active_groups:
                cov_pp, cov_nn = gsf_nucleon.p_and_n_covariance(
                    group1, group2, energies
                )
                cov_p_total += cov_pp
                cov_n_total += cov_nn
        
        # Extract errors (diagonal of covariance matrix)
        p_err_gsf = np.sqrt(np.diag(cov_p_total))
        n_err_gsf = np.sqrt(np.diag(cov_n_total))
        
        # Calculate relative errors for comparison
        _, p_flux_pca, n_flux_pca = pca_model.p_and_n_flux(energies)
        rel_err_p_pca = p_err_pca / p_flux_pca
        rel_err_p_gsf = p_err_gsf / p_flux_gsf
        rel_err_n_pca = n_err_pca / n_flux_pca
        rel_err_n_gsf = n_err_gsf / n_flux_gsf
        
        # Calculate ratios
        ratio_p = rel_err_p_pca / rel_err_p_gsf
        ratio_n = rel_err_n_pca / rel_err_n_gsf
        
        # Low-Rank Plus Diagonal exactly preserves diagonal variances, so errors
        # should be very close to GSF (within a few percent accounting for interpolation)
        # The diagonal term D ensures this preservation.
        
        # Check that proton ratios are within a few percent of 1.0
        assert np.all(ratio_p > 0.95), \
            f"Low-Rank Plus Diagonal proton errors too small: min ratio = {np.min(ratio_p):.3f}"
        assert np.all(ratio_p < 1.05), \
            f"Low-Rank Plus Diagonal proton errors too large: max ratio = {np.max(ratio_p):.3f}"
        
        # Check median is very close to 1.0 (within 2%)
        median_ratio_p = np.median(ratio_p)
        assert 0.98 < median_ratio_p < 1.02, \
            f"Low-Rank Plus Diagonal/GSF proton error ratio median {median_ratio_p:.3f} " \
            f"should be near 1.0 (exact diagonal preservation)"
        
        # Similar checks for neutron
        assert np.all(ratio_n > 0.95), \
            f"Low-Rank Plus Diagonal neutron errors too small: min ratio = {np.min(ratio_n):.3f}"
        assert np.all(ratio_n < 1.05), \
            f"Low-Rank Plus Diagonal neutron errors too large: max ratio = {np.max(ratio_n):.3f}"
        
        median_ratio_n = np.median(ratio_n)
        assert 0.98 < median_ratio_n < 1.02, \
            f"Low-Rank Plus Diagonal/GSF neutron error ratio median {median_ratio_n:.3f} " \
            f"should be near 1.0 (exact diagonal preservation)"

    def test_total_nucleon_error_comparison(self, pca_model, gsf_model, energies):
        """Compare total nucleon flux errors between Low-Rank Plus Diagonal and GSF2025.
        
        The Low-Rank Plus Diagonal approach exactly preserves diagonal variances.
        Total nucleon errors (p+n) are slightly higher (~85-125%) due to correlation effects.
        """
        # Get Low-Rank Plus Diagonal total error
        tot_err_pca = pca_model.tot_nucleon_flux_error(energies)
        tot_flux_pca = pca_model.tot_nucleon_flux(energies)
        
        # Get GSF nucleon model total error using the built-in method
        from globalsplinefit import GSFEnergyPerNucleon
        gsf_nucleon = GSFEnergyPerNucleon(version="2025")
        
        # Use GSF's total_error method which accounts for all groups and correlations
        tot_err_gsf = gsf_nucleon.total_error(energies)
        tot_flux_gsf = gsf_nucleon.total_flux(energies)
        
        # Calculate relative errors
        rel_err_pca = tot_err_pca / tot_flux_pca
        rel_err_gsf = tot_err_gsf / tot_flux_gsf
        
        # Check that errors are in reasonable range
        assert np.all(rel_err_pca > 0.0001), \
            f"Low-Rank Plus Diagonal errors unreasonably small: min = {np.min(rel_err_pca)*100:.3f}%"
        assert np.all(rel_err_pca < 0.5), \
            f"Low-Rank Plus Diagonal errors too large (>50%): max = {np.max(rel_err_pca)*100:.2f}%"
        
        # Compare Low-Rank Plus Diagonal and GSF errors
        # Low-Rank Plus Diagonal exactly preserves diagonal variances through D term.
        # Total nucleon errors can be slightly higher due to correlation effects.
        ratio = rel_err_pca / rel_err_gsf
        
        # Check ratio is reasonably close to 1.0 (allowing for correlation effects)
        assert np.all(ratio > 0.85), \
            f"Low-Rank Plus Diagonal total errors too small: min ratio = {np.min(ratio):.3f}"
        assert np.all(ratio < 1.25), \
            f"Low-Rank Plus Diagonal total errors too large: max ratio = {np.max(ratio):.3f}"
        
        # Check median ratio is close to 1.0 (within 10%)
        median_ratio = np.median(ratio)
        assert 0.90 < median_ratio < 1.15, \
            f"Low-Rank Plus Diagonal/GSF total error ratio median {median_ratio:.3f} " \
            f"should be near 1.0. " \
            f"Full range: [{np.min(ratio):.3f}, {np.max(ratio):.3f}]"

    def test_error_magnitude_reasonable(self, pca_model, energies):
        """Test that Low-Rank Plus Diagonal errors have reasonable magnitudes.
        
        The Low-Rank Plus Diagonal model exactly preserves the diagonal of the
        covariance matrix, so uncertainties should match the underlying GSF model
        within a few percent, typically in the range of 0.1-10% for most energies.
        """
        tot_err = pca_model.tot_nucleon_flux_error(energies)
        tot_flux = pca_model.tot_nucleon_flux(energies)
        
        rel_err = tot_err / tot_flux
        
        # Check that relative errors are in reasonable range
        # Low-Rank Plus Diagonal preserves GSF uncertainties (0.1-10%)
        assert np.all(rel_err > 0), "All errors should be positive"
        assert np.all(rel_err < 0.2), \
            f"Relative errors should be < 20% (preserves GSF errors). " \
            f"Got median: {np.median(rel_err)*100:.1f}%, " \
            f"max: {np.max(rel_err)*100:.1f}%"
        
        # Verify errors are in typical GSF range (most should be 0.1-10%)
        in_range = np.sum((rel_err > 0.001) & (rel_err < 0.10))
        assert in_range > 0.5 * len(energies), \
            f"Most relative errors should be 0.1-10% (matching GSF). " \
            f"Got median: {np.median(rel_err)*100:.1f}%, " \
            f"range: [{np.min(rel_err)*100:.1f}%, {np.max(rel_err)*100:.1f}%]"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
