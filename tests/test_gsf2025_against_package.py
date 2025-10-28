"""
Test suite to validate GlobalSplineFit2025 implementation against the
globalsplinefit package directly.

These tests ensure that the crflux wrapper correctly interfaces with the
underlying globalsplinefit models and produces consistent results.
"""

import pytest
import numpy as np

# Try to import both crflux and globalsplinefit
pytest.importorskip("globalsplinefit")

import crflux.models as crm
from globalsplinefit import GSFEnergy, GSFEnergyPerNucleon


class TestGlobalSplineFit2025AgainstPackage:
    """Test GlobalSplineFit2025 against direct globalsplinefit package calls."""

    @pytest.fixture
    def energies(self):
        """Common energy grid for testing."""
        return np.logspace(2, 10, 20)  # 100 GeV to 10^10 GeV

    @pytest.fixture
    def crflux_model(self):
        """crflux GlobalSplineFit2025 model instance."""
        return crm.GlobalSplineFit2025()

    @pytest.fixture
    def gsf_nucleus_model(self):
        """Direct GSFEnergy model instance."""
        return GSFEnergy()

    @pytest.fixture
    def gsf_nucleon_model(self):
        """Direct GSFEnergyPerNucleon model instance."""
        return GSFEnergyPerNucleon()

    def test_proton_flux_consistency(self, crflux_model, gsf_nucleus_model, energies):
        """Test that proton flux matches between implementations."""
        # Get proton flux from crflux (CORSIKA ID 14 for protons)
        crflux_proton_flux = crflux_model.nucleus_flux(14, energies)
        
        # Get proton flux directly from GSF (Z=1 for protons)
        gsf_proton_flux = gsf_nucleus_model.flux(energies, 1, time_interval=None)
        
        # They should be identical
        np.testing.assert_allclose(
            crflux_proton_flux,
            gsf_proton_flux,
            rtol=1e-10,
            err_msg="Proton flux differs between crflux and GSF implementations"
        )

    def test_helium_flux_consistency(self, crflux_model, gsf_nucleus_model, energies):
        """Test that helium flux matches between implementations."""
        # Get helium flux from crflux (CORSIKA ID 402 for He-4)
        crflux_helium_flux = crflux_model.nucleus_flux(402, energies)
        
        # Get helium flux directly from GSF (Z=2 for helium)
        gsf_helium_flux = gsf_nucleus_model.flux(energies, 2, time_interval=None)
        
        # They should be identical
        np.testing.assert_allclose(
            crflux_helium_flux,
            gsf_helium_flux,
            rtol=1e-10,
            err_msg="Helium flux differs between crflux and GSF implementations"
        )

    def test_oxygen_flux_consistency(self, crflux_model, gsf_nucleus_model, energies):
        """Test that oxygen flux matches between implementations."""
        # Get oxygen flux from crflux (CORSIKA ID 1608 for O-16)
        crflux_oxygen_flux = crflux_model.nucleus_flux(1608, energies)
        
        # Get oxygen flux directly from GSF (Z=8 for oxygen)
        gsf_oxygen_flux = gsf_nucleus_model.flux(energies, 8, time_interval=None)
        
        # They should be identical
        np.testing.assert_allclose(
            crflux_oxygen_flux,
            gsf_oxygen_flux,
            rtol=1e-10,
            err_msg="Oxygen flux differs between crflux and GSF implementations"
        )

    def test_iron_flux_consistency(self, crflux_model, gsf_nucleus_model, energies):
        """Test that iron flux matches between implementations."""
        # Get iron flux from crflux (CORSIKA ID 5626 for Fe-56)
        crflux_iron_flux = crflux_model.nucleus_flux(5626, energies)
        
        # Get iron flux directly from GSF (Z=26 for iron)
        gsf_iron_flux = gsf_nucleus_model.flux(energies, 26, time_interval=None)
        
        # They should be identical
        np.testing.assert_allclose(
            crflux_iron_flux,
            gsf_iron_flux,
            rtol=1e-10,
            err_msg="Iron flux differs between crflux and GSF implementations"
        )

    def test_total_flux_consistency(self, crflux_model, gsf_nucleus_model, energies):
        """Test that total flux matches between implementations."""
        # Get total flux from crflux
        crflux_total_flux = crflux_model.total_flux(energies)
        
        # Get total flux directly from GSF
        gsf_total_flux = gsf_nucleus_model.total_flux(energies, time_interval=None)
        
        # They should be identical
        np.testing.assert_allclose(
            crflux_total_flux,
            gsf_total_flux,
            rtol=1e-10,
            err_msg="Total flux differs between crflux and GSF implementations"
        )

    def test_p_and_n_flux_consistency(self, crflux_model, gsf_nucleon_model, energies):
        """Test that proton and neutron fluxes match between implementations."""
        # Get p and n flux from crflux
        crflux_p_frac, crflux_p_flux, crflux_n_flux = crflux_model.p_and_n_flux(energies)
        
        # Get p and n flux directly from GSF
        gsf_pn_flux = gsf_nucleon_model.p_and_n_total_flux(energies, time_interval=None)
        gsf_p_flux = gsf_pn_flux[0]
        gsf_n_flux = gsf_pn_flux[1]
        
        # Compare proton flux
        np.testing.assert_allclose(
            crflux_p_flux,
            gsf_p_flux,
            rtol=1e-10,
            err_msg="Proton flux differs between crflux and GSF implementations"
        )
        
        # Compare neutron flux
        np.testing.assert_allclose(
            crflux_n_flux,
            gsf_n_flux,
            rtol=1e-10,
            err_msg="Neutron flux differs between crflux and GSF implementations"
        )
        
        # Verify proton fraction calculation
        expected_p_frac = gsf_p_flux / (gsf_p_flux + gsf_n_flux)
        np.testing.assert_allclose(
            crflux_p_frac,
            expected_p_frac,
            rtol=1e-10,
            err_msg="Proton fraction differs from expected calculation"
        )

    def test_solar_modulation_LIS(self, energies):
        """Test that LIS (no solar modulation) works consistently."""
        # Create models with LIS
        crflux_model_lis = crm.GlobalSplineFit2025(time_interval="LIS")
        gsf_nucleus_model = GSFEnergy()
        
        # Get proton flux with LIS
        crflux_proton_lis = crflux_model_lis.nucleus_flux(14, energies)
        gsf_proton_lis = gsf_nucleus_model.flux(energies, 1, time_interval="LIS")
        
        # They should be identical
        np.testing.assert_allclose(
            crflux_proton_lis,
            gsf_proton_lis,
            rtol=1e-10,
            err_msg="LIS proton flux differs between implementations"
        )

    def test_solar_modulation_time_interval(self, energies):
        """Test that specific time intervals work consistently."""
        # Create models with specific time interval
        time_interval = (201501, 201512)  # 2015
        crflux_model_2015 = crm.GlobalSplineFit2025(time_interval=time_interval)
        gsf_nucleus_model = GSFEnergy()
        
        # Get proton flux for 2015
        crflux_proton_2015 = crflux_model_2015.nucleus_flux(14, energies)
        gsf_proton_2015 = gsf_nucleus_model.flux(
            energies, 1, time_interval=time_interval
        )
        
        # They should be identical
        np.testing.assert_allclose(
            crflux_proton_2015,
            gsf_proton_2015,
            rtol=1e-10,
            err_msg="Time-interval proton flux differs between implementations"
        )

    def test_version_parameter(self, energies):
        """Test that version parameter works consistently."""
        # Test with 2017 version
        crflux_model_2017 = crm.GlobalSplineFit2025(version="2017")
        gsf_nucleus_model_2017 = GSFEnergy(version="2017")
        
        # Get proton flux
        crflux_proton = crflux_model_2017.nucleus_flux(14, energies)
        gsf_proton = gsf_nucleus_model_2017.flux(energies, 1, time_interval=None)
        
        # They should be identical
        np.testing.assert_allclose(
            crflux_proton,
            gsf_proton,
            rtol=1e-10,
            err_msg="Version-specific proton flux differs between implementations"
        )

    def test_all_elements_consistency(self, crflux_model, gsf_nucleus_model, energies):
        """Test that all available elements produce consistent fluxes."""
        # Get list of all elements from GSF
        z_to_a = gsf_nucleus_model.z_to_a
        
        for z, a in z_to_a.items():
            # Convert to CORSIKA ID
            corsika_id = int(round(a)) * 100 + int(z)
            
            # Get flux from both implementations
            crflux_flux = crflux_model.nucleus_flux(corsika_id, energies)
            gsf_flux = gsf_nucleus_model.flux(energies, z, time_interval=None)
            
            # They should be identical
            np.testing.assert_allclose(
                crflux_flux,
                gsf_flux,
                rtol=1e-10,
                err_msg=f"Flux for Z={z} (CORSIKA ID {corsika_id}) differs"
            )

    def test_nucleus_ids_complete(self, crflux_model, gsf_nucleus_model):
        """Test that crflux nucleus_ids includes all elements from GSF."""
        # Get all Z values from GSF
        gsf_z_values = set(gsf_nucleus_model.z_to_a.keys())
        
        # Get all Z values from crflux nucleus_ids
        crflux_z_values = set()
        for corsika_id in crflux_model.nucleus_ids:
            z = corsika_id % 100
            crflux_z_values.add(z)
        
        # They should be the same
        assert (
            crflux_z_values == gsf_z_values
        ), f"crflux nucleus_ids missing elements: {gsf_z_values - crflux_z_values}"

    def test_scalar_and_array_input(self, crflux_model, gsf_nucleus_model):
        """Test that both scalar and array inputs work consistently."""
        # Scalar input
        scalar_energy = 1000.0
        crflux_scalar = crflux_model.nucleus_flux(14, scalar_energy)
        gsf_scalar = gsf_nucleus_model.flux(scalar_energy, 1, time_interval=None)
        
        np.testing.assert_allclose(
            crflux_scalar,
            gsf_scalar,
            rtol=1e-10,
            err_msg="Scalar input produces different results"
        )
        
        # Array input
        array_energy = np.array([100.0, 1000.0, 10000.0])
        crflux_array = crflux_model.nucleus_flux(14, array_energy)
        gsf_array = gsf_nucleus_model.flux(array_energy, 1, time_interval=None)
        
        np.testing.assert_allclose(
            crflux_array,
            gsf_array,
            rtol=1e-10,
            err_msg="Array input produces different results"
        )

    def test_lnA_calculation(self, crflux_model, gsf_nucleus_model, energies):
        """Test that lnA calculation uses consistent flux values."""
        # Calculate lnA from crflux
        crflux_lnA = crflux_model.lnA(energies)
        
        # Calculate lnA manually using GSF fluxes
        z_to_a = gsf_nucleus_model.z_to_a
        sum_weight = np.zeros_like(energies, dtype=float)
        
        for z, a in z_to_a.items():
            if z == 1:  # Skip protons (lnA = 0)
                continue
            flux = gsf_nucleus_model.flux(energies, z, time_interval=None)
            sum_weight += np.log(a) * flux
        
        total_flux = gsf_nucleus_model.total_flux(energies, time_interval=None)
        expected_lnA = sum_weight / total_flux
        
        # Allow small numerical differences (0.1% tolerance)
        np.testing.assert_allclose(
            crflux_lnA,
            expected_lnA,
            rtol=1e-3,
            err_msg="lnA calculation differs from expected"
        )

    def test_p_and_n_total_flux_method_exists(self, gsf_nucleon_model, energies):
        """Test that GSF's p_and_n_total_flux method is available and works."""
        # This test verifies that the method we're using in crflux exists
        # and returns the expected format
        pn_flux = gsf_nucleon_model.p_and_n_total_flux(energies, time_interval=None)
        
        # Should return array of shape (2, N) with positive fluxes
        assert pn_flux.shape == (2, len(energies)), "p_and_n_total_flux should return (2, N) array"
        assert np.all(pn_flux >= 0), "All fluxes should be non-negative"
        assert np.all(pn_flux[0] > 0), "Proton flux should be positive"
        assert np.all(pn_flux[1] > 0), "Neutron flux should be positive"


class TestGlobalSplineFit2017Version:
    """Test that GlobalSplineFit2025 with version='2017' matches GSF 2017 package."""

    @pytest.fixture
    def energies(self):
        """Common energy grid for testing."""
        return np.logspace(2, 8, 15)  # 100 GeV to 10^8 GeV

    @pytest.fixture
    def crflux_model_2017(self):
        """crflux GlobalSplineFit2025 model with 2017 version."""
        return crm.GlobalSplineFit2025(version="2017")

    @pytest.fixture
    def gsf_nucleus_model_2017(self):
        """Direct GSFEnergy model with 2017 version."""
        return GSFEnergy(version="2017")

    @pytest.fixture
    def gsf_nucleon_model_2017(self):
        """Direct GSFEnergyPerNucleon model with 2017 version."""
        return GSFEnergyPerNucleon(version="2017")

    def test_2017_proton_flux(self, crflux_model_2017, gsf_nucleus_model_2017, energies):
        """Test that 2017 proton flux matches exactly."""
        crflux_flux = crflux_model_2017.nucleus_flux(14, energies)
        gsf_flux = gsf_nucleus_model_2017.flux(energies, 1, time_interval=None)
        
        np.testing.assert_allclose(
            crflux_flux,
            gsf_flux,
            rtol=1e-10,
            err_msg="2017 proton flux differs between implementations"
        )

    def test_2017_helium_flux(self, crflux_model_2017, gsf_nucleus_model_2017, energies):
        """Test that 2017 helium flux matches exactly."""
        crflux_flux = crflux_model_2017.nucleus_flux(402, energies)
        gsf_flux = gsf_nucleus_model_2017.flux(energies, 2, time_interval=None)
        
        np.testing.assert_allclose(
            crflux_flux,
            gsf_flux,
            rtol=1e-10,
            err_msg="2017 helium flux differs between implementations"
        )

    def test_2017_iron_flux(self, crflux_model_2017, gsf_nucleus_model_2017, energies):
        """Test that 2017 iron flux matches exactly."""
        crflux_flux = crflux_model_2017.nucleus_flux(5626, energies)
        gsf_flux = gsf_nucleus_model_2017.flux(energies, 26, time_interval=None)
        
        np.testing.assert_allclose(
            crflux_flux,
            gsf_flux,
            rtol=1e-10,
            err_msg="2017 iron flux differs between implementations"
        )

    def test_2017_total_flux(self, crflux_model_2017, gsf_nucleus_model_2017, energies):
        """Test that 2017 total flux matches exactly."""
        crflux_flux = crflux_model_2017.total_flux(energies)
        gsf_flux = gsf_nucleus_model_2017.total_flux(energies, time_interval=None)
        
        np.testing.assert_allclose(
            crflux_flux,
            gsf_flux,
            rtol=1e-10,
            err_msg="2017 total flux differs between implementations"
        )

    def test_2017_nucleon_flux(self, crflux_model_2017, gsf_nucleon_model_2017, energies):
        """Test that 2017 proton and neutron fluxes match exactly."""
        crflux_p_frac, crflux_p_flux, crflux_n_flux = crflux_model_2017.p_and_n_flux(energies)
        
        gsf_pn_flux = gsf_nucleon_model_2017.p_and_n_total_flux(energies, time_interval=None)
        gsf_p_flux = gsf_pn_flux[0]
        gsf_n_flux = gsf_pn_flux[1]
        
        np.testing.assert_allclose(
            crflux_p_flux,
            gsf_p_flux,
            rtol=1e-10,
            err_msg="2017 proton flux differs between implementations"
        )
        
        np.testing.assert_allclose(
            crflux_n_flux,
            gsf_n_flux,
            rtol=1e-10,
            err_msg="2017 neutron flux differs between implementations"
        )

    def test_2017_lnA(self, crflux_model_2017, gsf_nucleus_model_2017, energies):
        """Test that 2017 lnA calculation is consistent."""
        crflux_lnA = crflux_model_2017.lnA(energies)
        
        # Calculate lnA manually using GSF 2017 fluxes
        z_to_a = gsf_nucleus_model_2017.z_to_a
        sum_weight = np.zeros_like(energies, dtype=float)
        
        for z, a in z_to_a.items():
            if z == 1:  # Skip protons (lnA = 0)
                continue
            flux = gsf_nucleus_model_2017.flux(energies, z, time_interval=None)
            sum_weight += np.log(a) * flux
        
        total_flux = gsf_nucleus_model_2017.total_flux(energies, time_interval=None)
        expected_lnA = sum_weight / total_flux
        
        np.testing.assert_allclose(
            crflux_lnA,
            expected_lnA,
            rtol=1e-3,
            err_msg="2017 lnA calculation differs from expected"
        )


class TestGlobalSplineFit2025EdgeCases:
    """Test edge cases and special conditions."""

    def test_very_low_energy(self):
        """Test behavior at very low energies."""
        crflux_model = crm.GlobalSplineFit2025()
        gsf_model = GSFEnergy()
        
        # Test at lower energy boundary
        E = 10.0  # 10 GeV
        crflux_flux = crflux_model.nucleus_flux(14, E)
        gsf_flux = gsf_model.flux(E, 1, time_interval=None)
        
        np.testing.assert_allclose(
            crflux_flux,
            gsf_flux,
            rtol=1e-10,
            err_msg="Low energy flux differs"
        )

    def test_very_high_energy(self):
        """Test behavior at very high energies."""
        crflux_model = crm.GlobalSplineFit2025()
        gsf_model = GSFEnergy()
        
        # Test at upper energy boundary
        E = 1e11  # 10^11 GeV
        crflux_flux = crflux_model.nucleus_flux(14, E)
        gsf_flux = gsf_model.flux(E, 1, time_interval=None)
        
        np.testing.assert_allclose(
            crflux_flux,
            gsf_flux,
            rtol=1e-10,
            err_msg="High energy flux differs"
        )

    def test_unavailable_nucleus(self):
        """Test that requesting an unavailable nucleus returns zero."""
        crflux_model = crm.GlobalSplineFit2025()
        
        # Request a nucleus that doesn't exist (e.g., element 99)
        fake_corsika_id = 25099  # A=250, Z=99
        E = 1000.0
        
        flux = crflux_model.nucleus_flux(fake_corsika_id, E)
        
        # Should return zero or array of zeros
        expected = np.zeros_like(np.atleast_1d(E))
        np.testing.assert_array_equal(
            np.atleast_1d(flux),
            expected,
            err_msg="Unavailable nucleus should return zero flux"
        )

    def test_zero_energy_handling(self):
        """Test that zero or negative energies are handled gracefully."""
        crflux_model = crm.GlobalSplineFit2025()
        
        # This might raise an error or return a specific value
        # depending on GSF implementation
        E = np.array([0.0, -1.0, 100.0])
        
        # Just ensure it doesn't crash - behavior may vary
        try:
            flux = crflux_model.nucleus_flux(14, E)
            # If it doesn't crash, verify the positive energy works
            assert flux[-1] > 0, "Positive energy should give positive flux"
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for invalid energies
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
