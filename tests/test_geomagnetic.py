import numpy as np
import pytest

import crflux.models as mods


class TestGeomagneticCutoff:
    def test_no_cutoff_by_default(self):
        mod = mods.HillasGaisser2012("H3a")
        assert mod.geomagnetic_cutoff is None

    def test_proton_cutoff(self):
        """Proton (Z=1): E_cut = 1 * 7 = 7 GeV."""
        mod = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
        below = mod.nucleus_flux(14, 5.0)
        above = mod.nucleus_flux(14, 10.0)
        assert float(below[0]) == 0.0
        assert float(above[0]) > 0.0

    def test_iron_cutoff(self):
        """Iron (Z=26, A=54): E_cut = 26 * 7 = 182 GeV."""
        mod = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
        below = mod.nucleus_flux(5426, 100.0)
        above = mod.nucleus_flux(5426, 200.0)
        assert float(below[0]) == 0.0
        assert float(above[0]) > 0.0

    def test_helium_cutoff(self):
        """Helium (Z=2): E_cut = 2 * 7 = 14 GeV."""
        mod = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
        below = mod.nucleus_flux(402, 10.0)
        above = mod.nucleus_flux(402, 20.0)
        assert float(below[0]) == 0.0
        assert float(above[0]) > 0.0

    def test_array_cutoff(self):
        """Cutoff should apply correctly to arrays."""
        mod = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
        E = np.array([1.0, 5.0, 7.0, 10.0, 100.0])
        result = mod.nucleus_flux(14, E)
        # Below 7 GeV should be zero
        assert result[0] == 0.0
        assert result[1] == 0.0
        # At and above 7 GeV should be nonzero
        assert result[2] > 0.0
        assert result[3] > 0.0
        assert result[4] > 0.0

    def test_cutoff_does_not_alter_above(self):
        """Flux above cutoff should be identical to model without cutoff."""
        mod = mods.HillasGaisser2012("H3a")
        mod_cut = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
        E = np.logspace(2, 6, 50)  # all well above 7 GeV
        np.testing.assert_array_equal(
            mod.nucleus_flux(14, E), mod_cut.nucleus_flux(14, E)
        )

    def test_total_flux_with_cutoff(self):
        mod_cut = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
        E = np.logspace(2, 6, 50)
        result = mod_cut.total_flux(E)
        assert np.all(result > 0)

    def test_p_and_n_flux_with_cutoff(self):
        mod_cut = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
        E = np.logspace(2, 6, 50)
        pfrac, p, n = mod_cut.p_and_n_flux(E)
        assert np.all(p > 0)
        assert np.all(n >= 0)

    def test_gsf_beta_cutoff(self):
        """GlobalSplineFitBeta applies cutoff to nucleon fluxes."""
        gsf = mods.GlobalSplineFitBeta(geomagnetic_cutoff=100.0)
        E = np.array([10.0, 50.0, 100.0, 200.0, 1000.0])
        _, p, n = gsf.p_and_n_flux(E)
        assert p[0] == 0.0
        assert p[1] == 0.0
        assert p[2] > 0.0  # at cutoff
        assert p[3] > 0.0


class TestCutoffAllModels:
    """Verify cutoff propagation through all model constructors."""

    MODELS_WITH_CUTOFF = [
        (mods.PolyGonato, (False,)),
        (mods.HillasGaisser2012, ("H3a",)),
        (mods.H3a_polygonato, ("H3a",)),
        (mods.GaisserStanevTilav, ("3-gen",)),
        (mods.CombinedGHandHG, ("H3a",)),
        (mods.ZatsepinSokolskaya, ("pamela",)),
        (mods.GaisserHonda, ()),
        (mods.Thunman, ()),
        (mods.SimplePowerlaw27, ()),
        (mods.GlobalSplineFitBeta, ()),
    ]

    @pytest.mark.parametrize(
        "cls,args",
        MODELS_WITH_CUTOFF,
        ids=[c.__name__ for c, _ in MODELS_WITH_CUTOFF],
    )
    def test_cutoff_stored(self, cls, args):
        mod = cls(*args, geomagnetic_cutoff=10.0)
        assert mod.geomagnetic_cutoff == 10.0
