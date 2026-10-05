import numpy as np
import pytest

import crflux.models as mods


class TestGeomagneticCutoff:
    def test_no_cutoff_by_default(self):
        mod = mods.HillasGaisser2012("H3a")
        assert mod.geomagnetic_cutoff is None

    @staticmethod
    def e_cut(Z, A, R):
        """Total energy of a nucleus (Z, A) at rigidity R."""
        return np.sqrt((Z * R) ** 2 + (A * mods.NUCLEON_MASS_GEV) ** 2)

    @pytest.mark.parametrize("cid", [14, 402, 1206, 2814, 5626])
    @pytest.mark.parametrize("R", [0.5, 3.0, 7.0, 12.0])
    def test_rigidity_threshold(self, cid, R):
        """Flux is zero exactly below p = Z R, i.e. E = sqrt((Z R)^2 + M^2)."""
        mod = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=R)
        Z, A = mod.Z_A(cid)
        e_cut = self.e_cut(Z, A, R)
        assert mod.cutoff_energy(Z, A, R) == pytest.approx(e_cut, rel=1e-15)
        E = e_cut * np.array([1 - 1e-9, 1 + 1e-9])
        below, above = mod.nucleus_flux(cid, E)
        assert below == 0.0
        assert above > 0.0
        p = np.sqrt(E**2 - (A * mods.NUCLEON_MASS_GEV) ** 2)
        assert p[0] / Z < R < p[1] / Z

    def test_ultrarelativistic_limit(self):
        """For E >> M the threshold reduces to E_cut = Z R."""
        for Z, A in [(1, 1), (2, 4), (26, 56)]:
            R = 1e4
            assert mods.PrimaryFlux.cutoff_energy(Z, A, R) == pytest.approx(
                Z * R, rel=1e-6
            )

    def test_threshold_above_old_step(self):
        """The rigidity threshold lies above the old E = Z R step."""
        mod = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=3.0)
        assert mod.nucleus_flux(14, 3.1)[0] == 0.0  # p = 2.95 GeV/c < 3 GV
        assert mod.nucleus_flux(402, 6.5)[0] == 0.0  # p = 5.0 GeV/c < 6 GV

    def test_zero_cutoff_is_no_cutoff(self):
        mod = mods.HillasGaisser2012("H3a")
        mod0 = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=0.0)
        E = np.logspace(-0.5, 3, 20)
        for cid in mod.nucleus_ids:
            np.testing.assert_array_equal(
                mod.nucleus_flux(cid, E), mod0.nucleus_flux(cid, E)
            )

    def test_array_cutoff(self):
        """Cutoff should apply correctly to arrays."""
        mod = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
        e_cut = self.e_cut(1, 1, 7.0)
        E = np.array([1.0, 5.0, 7.0, e_cut, 10.0, 100.0])
        result = mod.nucleus_flux(14, E)
        np.testing.assert_array_equal(result[:3], 0.0)
        assert np.all(result[3:] > 0.0)

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
        """GlobalSplineFitBeta cuts nucleons as free protons in rigidity."""
        with pytest.warns(DeprecationWarning, match="A/Z = 1"):
            gsf = mods.GlobalSplineFitBeta(geomagnetic_cutoff=100.0)
        e_cut = self.e_cut(1, 1, 100.0)
        E = np.array([10.0, 50.0, 100.0, e_cut, 200.0, 1000.0])
        _, p, n = gsf.p_and_n_flux(E)
        np.testing.assert_array_equal(p[:3], 0.0)
        np.testing.assert_array_equal(n[:3], 0.0)
        assert np.all(p[3:] > 0.0)


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
