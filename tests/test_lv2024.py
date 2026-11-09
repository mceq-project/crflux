import numpy as np
import pytest

import crflux.models as mods

# Points of the published curves (ApJ 979, 225; vector figures, Fig. 1 E^2.7 and
# Fig. 2 E^3 fluxes) where the printed tables reproduce them. Extraction:
# gsf-harness runs/2026-11-09_lv2024-published-check (reference_points.py).
FIG1_P_POP1 = [
    (62.082, 11185.0),
    (2526.6, 5286.2),
    (9950.7, 3224.2),
    (27468.0, 1497.4),
    (65109.0, 365.41),
]
FIG1_FE_POP2 = [
    (2.8881e7, 4563.9),
    (3.2168e7, 4601.1),
    (3.9011e7, 4590.8),
    (4.3147e7, 4586.6),
]
FIG2_ALL = [
    (3.8536e5, 2.1035e6),
    (1.3706e6, 2.8548e6),
    (3.2478e6, 3.4505e6),
    (5.1283e6, 3.5658e6),
]


@pytest.fixture
def lv():
    return mods.LvEtAl2024()


def test_published_tables(lv):
    """Table 3 and Table 4 of the published version (not arXiv v1)."""
    p = lv.params
    assert p["pop1"][14] == (5.4e-5, 2.88) and p["pop1"][402] == (5.7e-5, 2.75)
    assert p["ls"][14] == (2.0e-4, 2.77) and p["ls"][5626] == (2.5e-5, 2.45)
    assert p["pop2"][1206] == (1.0e-6, 2.40) and p["pop2"][12753] == (1.5e-8, 2.10)
    assert p["xg_le"] == {14: (8.0e-19, 2.67), 402: (8.5e-19, 2.67)}
    assert p["xg_he"] == {402: (3.1e-20, 2.0), 1407: (1.7e-20, 2.0)}
    assert (lv.ls_L_pc, lv.ls_tau_yr, lv.ls_D0, lv.ls_delta) == (500.0, 2e5, 5e27, 0.33)
    assert lv.nucleus_ids == [14, 402, 1206, 1407, 1608, 2010, 2412, 2814, 5626, 12753]


def test_equations(lv):
    """Eqs. 2-4 evaluated by hand for Fe (local source) and He (both XG parts)."""
    E = np.array([3e2, 1e4, 1e6, 1e9, 1e10])
    phi0, g = lv.params["ls"][5626]
    R = np.sqrt(E**2 - (56 * mods.NUCLEON_MASS_GEV) ** 2) / 26
    D = 5e27 * (R / 4) ** 0.33
    L, tau = 500 * 3.0856775814913673e18, 2e5 * 3.15576e7
    ls = (
        phi0
        * (E / 1e3) ** -g
        * np.exp(-E / (26 * 3.6e4))
        * np.exp(-(L**2) / (4 * D * tau))
    )
    np.testing.assert_allclose(lv.component_flux("ls", 5626, E), ls, rtol=1e-12)
    Ex = np.array([1e7, 1e8, 1e9, 1e10, 1e11])  # above the cut-in (no cosh overflow)
    xg = 8.5e-19 * (Ex / 1e8) ** -2.67 * np.exp(-Ex / 2e9) / np.cosh(2 * 1.9e7 / Ex)
    xg += 3.1e-20 * (Ex / 1e8) ** -2.0 * np.exp(-Ex / 8e9) / np.cosh(2 * 1.5e9 / Ex)
    np.testing.assert_allclose(
        lv.component_flux("xg_le", 402, Ex) + lv.component_flux("xg_he", 402, Ex),
        xg,
        rtol=1e-12,
    )
    total = sum(lv.component_flux(c, 5626, E) for c in lv.params)
    np.testing.assert_allclose(lv.nucleus_flux(5626, E), total, rtol=1e-14)


@pytest.mark.parametrize(
    ("component", "cid", "points", "power"),
    [("pop1", 14, FIG1_P_POP1, 2.7), ("pop2", 5626, FIG1_FE_POP2, 2.7)],
)
def test_published_figure_1(lv, component, cid, points, power):
    E, y = np.array(points).T
    np.testing.assert_allclose(
        E**power * lv.component_flux(component, cid, E), y, rtol=0.01
    )


def test_published_figure_2(lv):
    E, y = np.array(FIG2_ALL).T
    np.testing.assert_allclose(E**3 * lv.total_flux(E), y, rtol=0.01)


def test_below_rest_mass_is_zero(lv):
    assert (
        lv.nucleus_flux(12753, 100.0)[0] == 0.0 and lv.nucleus_flux(12753, 200.0)[0] > 0
    )
