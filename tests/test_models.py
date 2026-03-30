import numpy as np
import pytest

import crflux.models as mods


class TestZA:
    def test_proton(self):
        pf = mods.HillasGaisser2012("H3a")
        Z, A = pf.Z_A(14)
        assert Z == 1
        assert A == 1

    def test_helium(self):
        pf = mods.HillasGaisser2012("H3a")
        Z, A = pf.Z_A(402)
        assert Z == 2
        assert A == 4

    def test_carbon(self):
        pf = mods.HillasGaisser2012("H3a")
        Z, A = pf.Z_A(1206)
        assert Z == 6
        assert A == 12

    def test_iron(self):
        pf = mods.HillasGaisser2012("H3a")
        Z, A = pf.Z_A(5426)
        assert Z == 26
        assert A == 54

    def test_integer_A(self):
        """Z_A should return integer A, not float."""
        pf = mods.HillasGaisser2012("H3a")
        _, A = pf.Z_A(5426)
        assert isinstance(A, int)


class TestFindNearbyId:
    def test_exact_match(self):
        mod = mods.HillasGaisser2012("H3a")
        assert mod._find_nearby_id(14) == 14
        assert mod._find_nearby_id(5426) == 5426

    def test_nearby_match(self):
        mod = mods.HillasGaisser2012("H3a")
        # 1407 (N) is not in HG2012, closest is 1206 (C, A=12) or 2814 (Si)
        result = mod._find_nearby_id(1407)
        assert result in mod.nucleus_ids

    def test_too_far_raises(self):
        mod = mods.HillasGaisser2012("H3a")
        with pytest.raises(Exception, match="No similar nucleus found"):
            mod._find_nearby_id(20180)  # Hg, A=201


class TestNucleusFlux:
    def test_scalar_input(self, model):
        result = model.nucleus_flux(14, 1e4)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_array_input(self, model, energy_grid):
        result = model.nucleus_flux(14, energy_grid)
        assert result.shape == energy_grid.shape

    def test_positive_flux(self, model, energy_grid_short):
        result = model.nucleus_flux(14, energy_grid_short)
        assert np.all(result >= 0)

    def test_decreasing_with_energy(self, model):
        """Flux should generally decrease with energy."""
        E = np.array([1e3, 1e4, 1e5, 1e6])
        result = model.nucleus_flux(14, E)
        # Check that flux at 1e3 > flux at 1e6 (power law behavior)
        assert result[0] > result[-1]


class TestTotalFlux:
    def test_array_output(self, model, energy_grid_short):
        result = model.total_flux(energy_grid_short)
        assert result.shape == energy_grid_short.shape

    def test_positive(self, model, energy_grid_short):
        result = model.total_flux(energy_grid_short)
        assert np.all(result > 0)


class TestTotNucleonFlux:
    def test_array_output(self, model, energy_grid_short):
        result = model.tot_nucleon_flux(energy_grid_short)
        assert result.shape == energy_grid_short.shape

    def test_positive(self, model, energy_grid_short):
        result = model.tot_nucleon_flux(energy_grid_short)
        assert np.all(result > 0)


class TestPAndNFlux:
    def test_returns_three_tuple(self, model, energy_grid_short):
        result = model.p_and_n_flux(energy_grid_short)
        assert len(result) == 3

    def test_proton_fraction_range(self, model, energy_grid_short):
        pfrac, _, _ = model.p_and_n_flux(energy_grid_short)
        assert np.all(pfrac >= 0)
        assert np.all(pfrac <= 1)

    def test_positive_fluxes(self, model, energy_grid_short):
        _, p, n = model.p_and_n_flux(energy_grid_short)
        assert np.all(p >= 0)
        assert np.all(n >= 0)

    def test_shapes(self, model, energy_grid_short):
        pfrac, p, n = model.p_and_n_flux(energy_grid_short)
        assert pfrac.shape == energy_grid_short.shape
        assert p.shape == energy_grid_short.shape
        assert n.shape == energy_grid_short.shape


class TestLnA:
    def test_reasonable_range(self, model, energy_grid_short):
        result = model.lnA(energy_grid_short)
        # lnA should be between 0 (pure proton) and ~4 (pure iron, ln(56)~4)
        assert np.all(result >= -0.1)
        assert np.all(result <= 5.0)


class TestNucleonGamma:
    def test_negative_index(self, model):
        E = np.array([1e3, 1e4, 1e5])
        result = model.nucleon_gamma(E)
        # Spectral index should be negative (falling spectrum)
        assert np.all(result < 0)

    def test_typical_range(self, model):
        E = np.array([1e4])
        result = model.nucleon_gamma(E)
        # Typical cosmic ray spectral index is between -2 and -4
        assert np.all(result > -5)
        assert np.all(result < -1)


class TestNucleusGamma:
    def test_proton_index(self, model):
        E = np.array([1e4])
        result = model.nucleus_gamma(E, 14)
        assert np.all(result < 0)
        assert np.all(result > -5)


class TestDelta0:
    def test_range(self):
        mod = mods.HillasGaisser2012("H3a")
        E = np.logspace(2, 6, 20)
        result = mod.delta_0(E)
        # Proton excess should be in [-1, 1]
        assert np.all(result >= -1)
        assert np.all(result <= 1)


# Numerical regression tests at E = 1e4 GeV for proton flux
REGRESSION_DATA = [
    (mods.PolyGonato, (False,), 14, 1e4, 1.7021935349447246e-07),
    (mods.HillasGaisser2012, ("H3a",), 14, 1e4, 1.8506172134891679e-07),
    (mods.HillasGaisser2012, ("H4a",), 14, 1e4, 1.9259684487470443e-07),
    (mods.GaisserStanevTilav, ("3-gen",), 14, 1e4, 1.8863983153209037e-07),
    (mods.GaisserStanevTilav, ("4-gen",), 14, 1e4, 1.8843888749816914e-07),
    (mods.CombinedGHandHG, ("H3a",), 14, 1e4, 1.8506172134891679e-07),
    (mods.CombinedGHandHG, ("H4a",), 14, 1e4, 1.9259684487470443e-07),
    (mods.GaisserHonda, (), 14, 1e4, 1.633752512252613e-07),
    (mods.ZatsepinSokolskaya, ("pamela",), 14, 1e4, 1.3353e-07),
    (mods.ZatsepinSokolskaya, ("default",), 14, 1e4, 1.9852099999999998e-07),
    (mods.Thunman, (), 14, 1e4, 2.694318427183888e-07),
    (mods.SimplePowerlaw27, (), 14, 1e4, 2.694318427183888e-07),
]


@pytest.mark.parametrize(
    "cls,args,cid,energy,expected",
    REGRESSION_DATA,
    ids=[f"{c.__name__}({a[0] if a else ''})" for c, a, *_ in REGRESSION_DATA],
)
def test_regression(cls, args, cid, energy, expected):
    mod = cls(*args)
    result = mod.nucleus_flux(cid, energy)
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)


class TestGlobalSplineFitBeta:
    def test_instantiation(self):
        gsf = mods.GlobalSplineFitBeta()
        assert gsf.nucleus_ids == []

    def test_p_and_n_flux(self):
        gsf = mods.GlobalSplineFitBeta()
        E = np.logspace(1, 10, 100)
        pfrac, p, n = gsf.p_and_n_flux(E)
        assert p.shape == (100,)
        assert n.shape == (100,)
        assert np.all(p > 0)
        assert np.all(n > 0)

    def test_tot_nucleon_flux(self):
        gsf = mods.GlobalSplineFitBeta()
        E = np.logspace(1, 10, 100)
        result = gsf.tot_nucleon_flux(E)
        assert result.shape == (100,)
        assert np.all(result > 0)

    def test_nucleus_flux_returns_zeros(self):
        gsf = mods.GlobalSplineFitBeta()
        E = np.logspace(1, 6, 50)
        result = gsf.nucleus_flux(14, E)
        assert np.all(result == 0)


class TestModelVariantErrors:
    def test_hg_invalid_model(self):
        with pytest.raises(Exception, match="Unknown model"):
            mods.HillasGaisser2012("invalid")

    def test_gst_invalid_model(self):
        with pytest.raises(Exception, match="Unknown model"):
            mods.GaisserStanevTilav("invalid")

    def test_zs_invalid_model(self):
        with pytest.raises(Exception, match="Unknown model"):
            mods.ZatsepinSokolskaya("invalid")


class TestProtonOnlyModels:
    def test_thunman_non_proton(self):
        mod = mods.Thunman()
        result = mod.nucleus_flux(402, 1e4)
        assert np.all(result == 0)

    def test_simplepowerlaw_non_proton(self):
        mod = mods.SimplePowerlaw27()
        result = mod.nucleus_flux(402, 1e4)
        assert np.all(result == 0)
