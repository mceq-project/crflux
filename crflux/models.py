r"""
:mod:`models` --- models of the cosmic ray flux at the top of Earth's atmosphere
================================================================================

This module is a collection of models of the high-energy primary cosmic
ray flux, found in the literature of the last decades. The base class
:class:`PrimaryFlux` has various methods, which can be employed in lepton flux
calculations, cosmic ray physics, radiation physics etc.

The numbering scheme for nuclei is adapted from the Cosmic Ray Air-Shower Monte
Carlo `CORSIKA <https://web.ikp.kit.edu/corsika/>`_. Protons have the ID 14. The
composite ID for nuclei follows the formula :math:`ID=100 \cdot A + Z`, where
:math:`A` is the mass number and :math:`Z` the charge. With this scheme one can
easily obtain charge and mass from the ID and vice-versa (see :func:`PrimaryFlux.Z_A`).

The physics underlying each of the models can be found following the references
in this documentation.

.. note::

    As always, if you use one of the models for your work, please cite the corresponding
    publication!

Example::

    import crflux.models as mods
    import numpy as np

    model = mods.HillasGaisser2012("H3a")
    E = np.logspace(1, 11, 100)
    pfrac, p, n = model.p_and_n_flux(E)

    # With geomagnetic cutoff of 7 GV
    model_cut = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=7.0)
    pfrac_cut, p_cut, n_cut = model_cut.p_and_n_flux(E)
"""

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np

#: Mean nucleon mass in GeV, (m_p + m_n)/2, the value globalsplinefit uses.
#: The rest mass of a nucleus with mass number A is taken as A times this value.
NUCLEON_MASS_GEV = 0.93891872965

#: Former corsika ids, mapped to the current ones (iron was Fe-54 before 2.2.0).
LEGACY_IDS = {5426: 5626}


def _nucleon_only_message(model, what):
    return (
        f"{model.__class__.__name__} provides proton and neutron fluxes only, "
        f"no per-nucleus fluxes ({what}). Use crflux.models.GlobalSplineFit, "
        "the full per-species GSF model."
    )


def _get_closest(value, in_list):
    """Compares the parameters ``value`` with values of the
    iterable ``in_list`` and returns the index and value of closest
    element.

    Args:
      value (int,float): value to be looked for in the list
      in_list (list,tuple): list of values with which to compare
    Returns:
      tuple (int,float): (index, value) of closest value in the list
    """
    minindex = np.argmin(np.abs(in_list - value * np.ones(len(in_list))))
    return minindex, in_list[minindex]


class PrimaryFlux(metaclass=ABCMeta):
    """Base class for primary cosmic ray flux models.

    Args:
      geomagnetic_cutoff (float, optional): rigidity cutoff in GV. If set,
        the flux of each nucleus is zeroed below rigidity R_cut, i.e. where
        its momentum p = sqrt(E**2 - M**2) is below Z * R_cut. ``E`` is the
        total energy of the nucleus and M = A * :data:`NUCLEON_MASS_GEV` its
        rest mass, so the energy threshold is
        E_cut = sqrt((Z * R_cut)**2 + M**2), which tends to Z * R_cut for
        E >> M. ``None`` or 0 means no cutoff.
    """

    def __init__(self, geomagnetic_cutoff=None):
        self.nucleus_ids = None
        self.geomagnetic_cutoff = geomagnetic_cutoff

    @abstractmethod
    def _nucleus_flux(self, corsika_id, E):
        """Returns the flux of nuclei corresponding to
        the ``corsika_id`` at energy ``E``.

        Args:
          corsika_id (int): see :mod:`crflux` for description.
          E (numpy.ndarray): laboratory energy of nucleus in GeV
        Returns:
          (numpy.ndarray): flux of single nucleus type :math:`\\Phi_{nucleus}`
          in :math:`(\\text{m}^2 \\text{s sr GeV})^{-1}`
        """
        raise NotImplementedError(
            self.__class__.__name__ + "::_nucleus_flux(): Base class method called."
        )

    def nucleus_flux(self, corsika_id, E):
        """Returns the flux of nuclei corresponding to
        the ``corsika_id`` at energy ``E``, with optional geomagnetic cutoff.

        Args:
          corsika_id (int): see :mod:`crflux` for description.
          E (float or numpy.ndarray): laboratory energy of nucleus in GeV
        Returns:
          (numpy.ndarray): flux of single nucleus type :math:`\\Phi_{nucleus}`
          in :math:`(\\text{m}^2 \\text{s sr GeV})^{-1}`
        """
        E = np.atleast_1d(np.asarray(E, dtype=float))
        corsika_id = LEGACY_IDS.get(corsika_id, corsika_id)
        flux = self._nucleus_flux(corsika_id, E)
        if self.geomagnetic_cutoff:
            Z, A = self.Z_A(corsika_id)
            flux = np.where(
                E >= self.cutoff_energy(Z, A, self.geomagnetic_cutoff), flux, 0.0
            )
        return flux

    @staticmethod
    def cutoff_energy(Z, A, rigidity):
        """Total energy of a nucleus (Z, A) with rigidity ``rigidity``.

        Args:
          Z (int): charge
          A (int): mass number
          rigidity (float or numpy.ndarray): rigidity in GV
        Returns:
          (float or numpy.ndarray): total energy sqrt((Z R)^2 + M^2) in GeV,
          with M = A * :data:`NUCLEON_MASS_GEV`
        """
        return np.sqrt((Z * rigidity) ** 2 + (A * NUCLEON_MASS_GEV) ** 2)

    def total_flux(self, E):
        """Returns total flux of nuclei, the "all-particle-flux".

        Args:
          E (float or numpy.ndarray): laboratory energy of particles in GeV
        Returns:
          (numpy.ndarray): particle flux in :math:`\\Phi_{particles}` in
          :math:`(\\text{m}^2 \\text{s sr GeV})^{-1}`
        """
        self._require_nuclei("total_flux")
        return sum(self.nucleus_flux(corsika_id, E) for corsika_id in self.nucleus_ids)

    def tot_nucleon_flux(self, E):
        """Returns total flux of nucleons, the "all-nucleon-flux".

        Args:
          E (float or numpy.ndarray): laboratory energy of nucleons in GeV
        Returns:
          (numpy.ndarray): nucleon flux :math:`\\Phi_{nucleons}` in
          :math:`(\\text{m}^2 \\text{s sr GeV})^{-1}`
        """
        return sum(
            self.Z_A(corsika_id)[1] ** 2.0
            * self.nucleus_flux(corsika_id, E * self.Z_A(corsika_id)[1])
            for corsika_id in self.nucleus_ids
        )

    def nucleon_gamma(self, E, rel_delta=0.01):
        """Returns differential spectral index of all-nucleon-flux
        obtained from a numerical derivative.

        Args:
          E (float or numpy.ndarray): laboratory energy of nucleons in GeV
          rel_delta (float): range of derivative relative to log10(E)
        Returns:
          (numpy.ndarray): spectral index :math:`\\gamma` of nucleons
        """
        delta = rel_delta * E
        fl = self.tot_nucleon_flux
        return np.log10(fl(E + delta) / fl(E - delta)) / np.log10(
            (E + delta) / (E - delta)
        )

    def nucleus_gamma(self, E, corsika_id, rel_delta=0.01):
        """Returns differential spectral index of nuclei
        obtained from a numerical derivative.

        Args:
          E (float or numpy.ndarray): laboratory energy of nuclei in GeV
          corsika_id (int): corsika id of nucleus/mass group
          rel_delta (float): range of derivative relative to log10(E)
        Returns:
          (numpy.ndarray): spectral index :math:`\\gamma` of nuclei
        """
        delta = rel_delta * E
        return np.log10(
            self.nucleus_flux(corsika_id, E + delta)
            / self.nucleus_flux(corsika_id, E - delta)
        ) / np.log10((E + delta) / (E - delta))

    def delta_0(self, E):
        """Returns proton excess.

        The proton excess is defined as
        :math:`\\delta_0 = \\frac{\\Phi_p - \\Phi_n}{\\Phi_p + \\Phi_n}`,
        with the nucleon fluxes from :meth:`p_and_n_flux`.

        Args:
          E (float or numpy.ndarray): laboratory energy of nucleons in GeV
        Returns:
          (numpy.ndarray): proton excess :math:`\\delta_0`
        """
        _, p_0, n_0 = self.p_and_n_flux(E)
        total = p_0 + n_0
        return np.divide(
            p_0 - n_0, total, out=np.zeros_like(total, dtype=float), where=total > 0
        )

    def p_and_n_flux(self, E):
        """Returns tuple with proton fraction, proton flux and neutron flux.

        The proton fraction is defined as :math:`\\frac{\\Phi_p}{\\Phi_p + \\Phi_n}`.

        Args:
          E (float or numpy.ndarray): laboratory energy of nucleons in GeV
        Returns:
          (float,float,float): proton fraction, proton flux, neutron flux
        """
        za = self.Z_A

        p_flux = sum(
            za(corsika_id)[0]
            * za(corsika_id)[1]
            * self.nucleus_flux(corsika_id, E * za(corsika_id)[1])
            for corsika_id in self.nucleus_ids
        )

        n_flux = sum(
            (za(corsika_id)[1] - za(corsika_id)[0])
            * za(corsika_id)[1]
            * self.nucleus_flux(corsika_id, E * za(corsika_id)[1])
            for corsika_id in self.nucleus_ids
        )

        total = p_flux + n_flux
        p_frac = np.where(total > 0, p_flux / total, 0.0)
        return p_frac, p_flux, n_flux

    def lnA(self, E):
        """Returns mean logarithmic mass <ln A>.

        Args:
          E (float or numpy.ndarray): laboratory energy of particles in GeV
        Returns:
          (numpy.ndarray): mean (natural) logarithmic mass
        """
        self._require_nuclei("lnA")
        sum_weight = 0.0

        for cid in self.nucleus_ids:
            if cid == 14:
                continue  # p has lnA = 0
            sum_weight += np.log(self.Z_A(cid)[1]) * self.nucleus_flux(cid, E)

        return sum_weight / self.total_flux(E)

    def _require_nuclei(self, method):
        """Raise for nucleon-only models, which have no per-nucleus fluxes."""
        if not self.nucleus_ids:
            raise NotImplementedError(_nucleon_only_message(self, method))

    def _find_nearby_id(self, corsika_id, delta_A=3):
        """Looks in :attr:`self.params` for a nucleus with same ``corsika_id``
        and returns the corsika_id if these parameters exist. If not, it will
        look for nuclei of mass number +- delta_A around the requested nucleus
        and return its corsika_id if it exists.

        Args:
          corsika_id (int): corsika id of nucleus/mass group
          delta_A (int): maximum mass number difference to accept
        Returns:
          (int): corsika_id of requested or similar nucleus
        Raises:
          Exception: if no nucleus with mass number closer than delta_A can
            be found in parameters
        """
        corsika_id = LEGACY_IDS.get(corsika_id, corsika_id)
        if corsika_id in self.nucleus_ids:
            return corsika_id

        A_in = (corsika_id - corsika_id % 100) // 100
        closest_id = _get_closest(corsika_id, np.array(self.nucleus_ids))[1]
        A_close = (closest_id - closest_id % 100) // 100
        if np.abs(A_in - A_close) > delta_A:
            e = (
                "{0}::_find_nearby_id(): No similar nucleus found with "
                + "delta_A <= {1} for A_in = {2}. Closest is {3}."
            )
            raise Exception(e.format(self.__class__.__name__, delta_A, A_in, A_close))
        else:
            return closest_id

    def Z_A(self, corsika_id):
        """Returns charge :math:`Z` and mass number :math:`A` corresponding
        to ``corsika_id``

        Args:
          corsika_id (int): corsika id of nucleus/mass group
        Returns:
          (int,int): (Z, A) tuple
        """
        Z, A = 1, 1
        if corsika_id > 14:
            Z = corsika_id % 100
            A = (corsika_id - Z) // 100
        return Z, A


class PolyGonato(PrimaryFlux):
    """J. R. Hoerandel, Astroparticle Physics 19, 193 (2003)."""

    def __init__(self, constdelta=False, **kwargs):
        super().__init__(**kwargs)

        self.name = "poly-gonato"
        self.sname = "pg"
        self.constdelta = constdelta
        self.E_p = 4.51e6
        self.gamma_c = -4.68
        self.epsilon_c = 1.87
        self.delta_gamma = 2.10
        if self.constdelta:
            self.epsilon_c = 1.90
            self.E_p = 4.49e6

        self.params = {}

        self.params[14] = (8.73e-2, 2.71, 1)  # H
        self.params[402] = (5.71e-2, 2.64, 2)  # He
        self.params[1206] = (1.06e-2, 2.66, 6)  # C
        self.params[1407] = (2.35e-3, 2.72, 7)  # N
        self.params[1608] = (1.57e-2, 2.68, 8)  # O
        self.params[2412] = (8.01e-3, 2.64, 12)  # Mg
        self.params[2613] = (1.15e-3, 2.66, 13)  # Al
        self.params[2814] = (7.96e-3, 2.75, 14)  # Si
        self.params[5025] = (1.35e-3, 2.46, 25)  # Mn
        self.params[5626] = (2.04e-2, 2.59, 26)  # Fe
        self.params[5427] = (7.51e-5, 2.72, 27)  # Co

        self.nucleus_ids = list(self.params.keys())

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)
        return self._polygonato_formula(corsika_id, E)

    def _polygonato_formula(self, corsika_id, E):
        p = self.params[corsika_id]
        gam = (self.gamma_c + p[1]) if self.constdelta else -self.delta_gamma

        return (
            p[0]
            / 1000.0
            * (E / 1000.0) ** (-p[1])
            * (1 + (E / p[2] / self.E_p) ** self.epsilon_c) ** (gam / self.epsilon_c)
        )


class _BenzviMontaruli(PrimaryFlux):
    """http://wiki.icecube.wisc.edu/index.php/Composition
    Project started by Segev BenZvi and T. Montaruli but now it
    is not maintained and rather canceled. Feb. 2014
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            **{k: v for k, v in kwargs.items() if k == "geomagnetic_cutoff"}
        )
        self.name = "Benzvi-Montaruli"
        self.sname = "BM"
        self.params = {}
        # dictionary[corsika_id] = (E_0, A, gamma1, Eb, gamma2)
        self.params[14] = (31.623, 1.145, -2.786, 262.628, -2.695)  # H
        self.params[402] = (100.0, 0.030, -2.712, 484.059, -2.600)  # He
        self.params[1206] = (240.0, 6.014e-4, -2.741, 2400.0, -2.503)  # C
        self.params[1608] = (320.0, 3.828e-5, -2.741, 3200.0, -2.503)  # O
        self.params[2412] = (400.0, 4.458e-5, -2.741, 4800.0, -2.503)  # Mg
        self.params[2814] = (560.0, 4.314e-5, -2.741, 5600.0, -2.503)  # Si
        self.params[5626] = (1120.0, 1.659e-5, -2.741, 11200.0, -2.503)  # Fe

        self.nucleus_ids = list(self.params.keys())

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)
        return self._benzvi_flux(corsika_id, E)

    def _benzvi_flux(self, corsika_id, E):
        param = self.params[corsika_id]
        E = np.atleast_1d(np.asarray(E, dtype=float))
        result = np.empty_like(E)
        lo = E < param[3]
        hi = ~lo
        result[lo] = param[1] * (E[lo] / param[0]) ** param[2]
        result[hi] = (
            param[1]
            * (param[3] / param[0]) ** (param[2] - param[4])
            * (E[hi] / param[0]) ** param[4]
        )
        return result


class HillasGaisser2012(PrimaryFlux):
    """Gaisser, T.K., Astroparticle Physics 35, 801 (2012).

    Model is based on Hillas ideas and eye-ball fits by T.K. Gaisser.
    H3a is a 3-'peters cycle' 5 mass group model with mixed
    composition above the ankle. H4a has protons only in the
    3. component.

    Args:
      model (str): can be either H3a or H4a.
    """

    def __init__(self, model="H4a", **kwargs):
        super().__init__(**kwargs)

        self.name = "Hillas-Gaisser (" + model + ")"
        self.sname = model
        self.model = model
        self.params = {}
        self.rid_cutoff = {}

        mass_comp = [14, 402, 1206, 2814, 5626]
        for mcomp in mass_comp:
            self.params[mcomp] = {}

        self.rid_cutoff[1] = 4e6
        self.rid_cutoff[2] = 30e6
        self.rid_cutoff[3] = 2e9
        self.params[14][1] = (7860, 1.66, 1)  # H
        self.params[402][1] = (3550, 1.58, 2)  # He
        self.params[1206][1] = (2200, 1.63, 6)  # CNO
        self.params[2814][1] = (1430, 1.67, 14)  # MgAlSi
        self.params[5626][1] = (2120, 1.63, 26)  # Fe

        self.params[14][2] = (20, 1.4, 1)  # H
        self.params[402][2] = (20, 1.4, 2)  # He
        self.params[1206][2] = (13.4, 1.4, 6)  # CNO
        self.params[2814][2] = (13.4, 1.4, 14)  # MgAlSi
        self.params[5626][2] = (13.4, 1.4, 26)  # Fe

        if self.model == "H3a":
            self.rid_cutoff[3] = 2e9
            self.params[14][3] = (1.7, 1.4, 1)  # H
            self.params[402][3] = (1.7, 1.4, 2)  # He
            self.params[1206][3] = (1.14, 1.4, 6)  # CNO
            self.params[2814][3] = (1.14, 1.4, 14)  # MgAlSi
            self.params[5626][3] = (1.14, 1.4, 26)  # Fe
        elif self.model == "H4a":
            self.rid_cutoff[3] = 60e9
            self.params[14][3] = (200.0, 1.6, 1)  # H
            self.params[402][3] = (0, 1.4, 2)  # He
            self.params[1206][3] = (0, 1.4, 6)  # CNO
            self.params[2814][3] = (0, 1.4, 14)  # MgAlSi
            self.params[5626][3] = (0, 1.4, 26)  # Fe
        else:
            raise Exception("HillasGaisser2012(): Unknown model version requested.")

        self.nucleus_ids = list(self.params.keys())

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        flux = 0.0
        for i in range(1, 4):
            p = self.params[corsika_id][i]
            flux += p[0] * E ** (-p[1] - 1.0) * np.exp(-E / p[2] / self.rid_cutoff[i])
        return flux


class H3a_polygonato(HillasGaisser2012):
    """Modified version of Gaisser, T.K., Astroparticle
    Physics 35, 801 (2012).

    Model is based on Hillas ideas and eye-ball fits by T.K. Gaisser.
    H3a is a 3-'peters cycle' 5 mass group model with mixed
    composition above the ankle. H4a has protons only in the
    3. component.

    This version has a five component poly-gonato at lower energies
    and resembles H3/4a at energies above the knee.

    Args:
      model (str): can be either H3a or H4a.
    """

    def __init__(self, model="H3a", **kwargs):
        super().__init__(model, **kwargs)
        self.rid_cutoff[1] = 4.49e6
        self.rid_cutoff[2] = 30e6
        self.rid_cutoff[3] = 2e9
        self.params[14][1] = (11800, 1.71, 1)  # H
        self.params[402][1] = (4750, 1.64, 2)  # He
        self.params[1206][1] = (3860, 1.67, 6)  # CNO
        self.params[2814][1] = (3120, 1.70, 14)  # MgAlSi
        self.params[5626][1] = (1080, 1.55, 26)  # Fe

        self.params[14][2] = (11.8, 1.4, 1)  # H
        self.params[402][2] = (11.8, 1.4, 2)  # He
        self.params[1206][2] = (7.88, 1.4, 6)  # CNO
        self.params[2814][2] = (7.88, 1.4, 14)  # MgAlSi
        self.params[5626][2] = (7.88, 1.4, 26)  # Fe


class GaisserStanevTilav(PrimaryFlux):
    """T. K. Gaisser, T. Stanev, and S. Tilav, arXiv:1303.3565, (2013).

    Args:
      model (str): 3-gen or 4-gen
      include_heavy_in_total (bool, optional): if True, the total flux will
        include the heavy component groups (Te, Hg) as well. Default is
        False for legacy compatibility.

    Raises:
      Exception: if ``model`` not properly specified.
    """

    def __init__(self, model="3-gen", include_heavy_in_total=False, **kwargs):
        super().__init__(**kwargs)

        self.name = "GST (" + model + ")"
        self.sname = "GST" + model[0]
        self.model = model
        self.params = {}
        self.rid_cutoff = {}

        self.rid_cutoff[1] = 120e3
        self.rid_cutoff[2] = 4e6

        mass_comp = [14, 402, 1206, 1608, 5626, 12852, 20180]
        for mcomp in mass_comp:
            self.params[mcomp] = {}

        self.params[14][1] = (7000, 1.66, 1)  # H
        self.params[402][1] = (3200, 1.58, 2)  # He
        self.params[1206][1] = (100, 1.4, 6)  # C
        self.params[1608][1] = (130, 1.4, 8)  # O
        self.params[5626][1] = (60, 1.3, 26)  # Fe
        self.params[12852][1] = (0, 1.0, 52)  # Te
        self.params[20180][1] = (0, 1.0, 80)  # Hg

        self.params[14][2] = (150, 1.4, 1)  # H
        self.params[402][2] = (65, 1.3, 2)  # He
        self.params[1206][2] = (6, 1.3, 6)  # C
        self.params[1608][2] = (7, 1.3, 8)  # O

        if self.model == "3-gen":
            self.params[5626][2] = (2.3, 1.2, 26)  # Fe
            self.params[12852][2] = (0.1, 1.2, 52)  # Te
            self.params[20180][2] = (0.4, 1.2, 80)  # Hg
            self.rid_cutoff[3] = 1.3e9

            self.params[14][3] = (14, 1.4, 1)  # H
            self.params[402][3] = (0, 1.4, 2)  # He
            self.params[1206][3] = (0, 1.4, 6)  # CNO
            self.params[1608][3] = (0, 1.3, 8)  # O
            self.params[5626][3] = (0.025, 1.2, 26)  # Fe
            self.params[12852][3] = (0, 1.0, 52)  # Te
            self.params[20180][3] = (0, 1.0, 80)  # Hg

        elif self.model == "4-gen":
            self.params[5626][2] = (2.1, 1.2, 26)  # Fe
            self.params[12852][2] = (0.1, 1.2, 52)  # Te
            self.params[20180][2] = (0.53, 1.2, 80)  # Hg

            self.rid_cutoff[3] = 1.5e9
            self.params[14][3] = (12.0, 1.4, 1)  # H
            self.params[402][3] = (0, 1.4, 2)  # He
            self.params[1206][3] = (0, 1.4, 6)  # CNO
            self.params[1608][3] = (0, 1.3, 8)  # O
            self.params[5626][3] = (0.011, 1.2, 26)  # Fe
            self.params[12852][3] = (0, 1.0, 52)  # Te
            self.params[20180][3] = (0, 1.0, 80)  # Hg

            self.rid_cutoff[4] = 40e9
            self.params[14][4] = (1.2, 1.4, 1)  # H
            self.params[402][4] = (0, 0, 2)  # He
            self.params[1206][4] = (0, 0, 6)  # CNO
            self.params[1608][4] = (0, 0, 8)  # O
            self.params[5626][4] = (0, 0, 26)  # Fe
            self.params[12852][4] = (0, 1.0, 52)  # Te
            self.params[20180][4] = (0, 1.0, 80)  # Hg
        else:
            raise Exception("GaisserStanevTilav(): Unknown model version.")
        if include_heavy_in_total:
            self.nucleus_ids = list(self.params.keys())
        else:
            self.nucleus_ids = [
                k for k in self.params.keys() if k not in [12852, 20180]
            ]

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        flux = 0.0
        ngen = 0

        if self.model == "3-gen":
            ngen = 4
        elif self.model == "4-gen":
            ngen = 5
        else:
            raise Exception("GaisserStanevTilav(): Unknown model type.")

        for i in range(1, ngen):
            p = self.params[corsika_id][i]
            flux += p[0] * E ** (-p[1] - 1.0) * np.exp(-E / p[2] / self.rid_cutoff[i])
        return flux


class CombinedGHandHG(PrimaryFlux):
    """A. Fedynitch, J. Becker Tjus, and P. Desiati,
    Phys. Rev. D 86, 114024 (2012).

    In the paper the high energy models were called cHGm for GH+H3a
    and cHGp for GH+H4a. The names have been changed to use the quite
    unintuitive names H3a and H4a in ongoing literature.
    """

    def __init__(self, model="H3a", **kwargs):
        super().__init__(**kwargs)
        self.name = "comb. GH and " + model
        self.sname = "c" + model
        self.params = {}
        self.leModel = GaisserHonda(**kwargs)
        # PRD 86, 114024 ran GH on the HG groups: CNO as 1206 and Mg-Si as 2814,
        # the latter with alpha + 1 sigma and K - 1 sigma
        gh = self.leModel.params
        gh[1206] = gh.pop(1407)
        del gh[2512]
        gh[2814] = (2.79 + 0.08, 34.2 - 6.0, 2.14, 0.01)
        self.leModel.nucleus_ids = list(gh)
        self.heModel = HillasGaisser2012(model, **kwargs)
        self.heCutOff = 1e5
        cid_list = [14, 402, 1206, 2814, 5626]

        # Store low- to high-energy model transitions in params
        for cid in cid_list:
            self.params[cid] = self.find_transition(cid)
        self.nucleus_ids = list(self.params.keys())

    def find_transition(self, corsika_id):
        from scipy.optimize import fsolve

        def func(logE):
            return self.leModel._nucleus_flux(
                corsika_id, np.atleast_1d(10**logE)
            ) - self.heModel._nucleus_flux(corsika_id, np.atleast_1d(10**logE))

        result = fsolve(func, 3.1)
        return 10 ** result[0]

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)
        E = np.atleast_1d(np.asarray(E, dtype=float))
        result = np.empty_like(E)
        lo = E < self.params[corsika_id]
        hi = ~lo
        if np.any(lo):
            result[lo] = self.leModel._nucleus_flux(corsika_id, E[lo])
        if np.any(hi):
            result[hi] = self.heModel._nucleus_flux(corsika_id, E[hi])
        return result


class ZatsepinSokolskaya(PrimaryFlux):
    """The model was first released in V. I. Zatsepin and N. V. Sokolskaya,
    Astronomy and Astrophysics 458, 1 (2006).

    Later, the PAMELA experiment has fitted the parameters of this model
    to their data in PAMELA Collaboration, O. Adriani et al.,
    Science 332, 69 (2011). Both versions of parameters can be accessed here.

    The model does not describe the flux above the knee. Therefore, the
    highest energies should not exceed 1-10 PeV.

    The paper's spectra are functions of the kinetic energy T of the nucleus
    (Eqs. 7-8): R = sqrt(T^2 + 2 A m_p T) / Z, and solar modulation in the
    force-field form with Phi = 600 MV, flux(T + Z Phi) P(T). crflux
    evaluates them at T = E - A m_p for the total energy E (dT/dE = 1) and
    applies the modulation at all energies.

    Args:
      model (str): 'default' for original or 'pamela' for PAMELA parameters
    """

    #: Proton mass in GeV (m_p in Eqs. 7-8 of the paper).
    m_p = 0.938272
    #: Force-field modulation potential in GV (Phi = 600 MV in the paper).
    phi_mod = 0.6

    def __init__(self, model="pamela", **kwargs):
        super().__init__(**kwargs)
        if model == "pamela":
            self.name = "Zatsepin-Sokolskaya/Pamela"
            self.sname = "ZSP"
            self.R_0 = 5.5
            self.alpha = (2.3, 2.1, 2.57)
            self.R_max = (8e4, 4e6, 2e2)
            self.gamma = (2.63, 2.43, 2.9)
            self.gamma_k = (8, 4.5, 4.5)
            self.f_norm = {}
            self.f_norm[14] = (7.1e3, 6.25e3, 3.0, 74.0, 1)
            self.f_norm[402] = (9.5e3, 8.5e3, 0.74, 18.0, 2)
            self.f_norm[1206] = (6.75e3, 1.8e3, 30, 5.8, 7)
            self.f_norm[2814] = (5.5e3, 1.5e3, 110, 3.5, 12)
            self.f_norm[5626] = (3.5e3, 1.2e3, 750, 2.4, 20)
        elif model == "default":
            self.name = "Zatsepin-Sokolskaya"
            self.sname = "ZS"
            self.R_0 = 5.5
            self.alpha = (2.3, 2.1, 2.57)
            self.R_max = (8e4, 4e6, 2e2)
            self.gamma = (2.63, 2.43, 2.9)
            self.gamma_k = (8, 4.5, 4.5)
            self.f_norm = {}
            self.f_norm[14] = (1.36e4, 6.25e3, 2.1, 74.0, 1)
            self.f_norm[402] = (8.75e3, 8.5e3, 3.0, 18.0, 2)
            self.f_norm[1206] = (6.75e3, 1.8e3, 30, 5.8, 7)
            self.f_norm[2814] = (5.5e3, 1.5e3, 110, 3.5, 12)
            self.f_norm[5626] = (3.5e3, 1.2e3, 750, 2.4, 20)
        else:
            raise Exception(
                f"{self.__class__.__name__}():: Unknown model selection '{model}'."
            )
        self.nucleus_ids = list(self.f_norm.keys())

    def lamba_esc(self, R):
        return (
            4.2 * (R / self.R_0) ** (-1.0 / 3.0) * (1 + (R / self.R_0) ** (-2.0 / 3.0))
        )

    def Q(self, R, gen):
        return R ** (-self.alpha[gen]) * self.phi(R, gen)

    def phi(self, R, gen):
        return (1 + (R / self.R_max[gen]) ** 2) ** (
            (self.gamma[gen] - self.gamma_k[gen]) / 2.0
        )

    def dR_dE(self, E, corsika_id):
        Z, A = self.Z_A(corsika_id)
        return 1.0 / Z * (E + self.m_p * A) / np.sqrt(E**2 + 2 * self.m_p * A * E)

    def R(self, E, corsika_id):
        Z, A = self.Z_A(corsika_id)
        return 1.0 / Z * np.sqrt(E**2 + 2 * self.m_p * A * E)

    def f_mod(self, E, corsika_id):
        """Force-field modulated flux at kinetic energy ``E`` of the nucleus."""
        Z, A = self.Z_A(corsika_id)
        M = A * self.m_p
        E_is = E + Z * self.phi_mod
        P = (E**2 + 2 * M * E) / (E_is**2 + 2 * M * E_is)
        return self.flux(E_is, corsika_id) * P

    def dN_dR(self, R, corsika_id, gen):
        return (
            self.Q(R, gen)
            * self.lamba_esc(R)
            / (1 + self.lamba_esc(R) / self.f_norm[corsika_id][3])
        )

    def dN_dE(self, E, corsika_id, gen):
        return self.dR_dE(E, corsika_id) * self.dN_dR(
            self.R(E, corsika_id), corsika_id, gen
        )

    def flux(self, E, corsika_id):
        flux = 0.0
        for gen in range(3):
            flux += (
                self.f_norm[corsika_id][gen]
                * 1e4 ** (-2.75)
                / self.dN_dE(1e4, corsika_id, gen)
                * self.dN_dE(E, corsika_id, gen)
            )
        return flux

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)
        E = np.atleast_1d(np.asarray(E, dtype=float))
        T = E - self.Z_A(corsika_id)[1] * self.m_p
        result = np.zeros_like(E)
        above = T > 0
        if np.any(above):
            result[above] = self.f_mod(T[above], corsika_id)
        return result


class GaisserHonda(PrimaryFlux):
    """5 mass group single power-law model from T.K. Gaisser and M. Honda,
    Annual Review of Nuclear and Particle Science 52, 153 (2002)

    This model was tuned to lower energy baloon data. It fails to
    describe the flux at and above the knee. A safe range for using
    it is < 100 TeV/nucleon.

    The fit phi(E_k) = K (E_k + b exp(-c sqrt(E_k)))^-alpha (Eq. 10 and
    Table 2 of the paper, central values, He "high"; Honda et al., PRD 70,
    043008 (2004), Eq. 1) gives the nuclei per GeV of kinetic energy per
    nucleon E_k. The CNO (A = 14) and Mg-Si (A = 25) groups are represented by
    14N (1407) and 25Mg (2512). crflux evaluates it
    at E_k = E/A - :data:`NUCLEON_MASS_GEV` for the total energy E of the
    nucleus and divides by A (dE_k/dE = 1/A).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(geomagnetic_cutoff=kwargs.pop("geomagnetic_cutoff", None))
        self.name = "Gaisser-Honda"
        self.sname = "GH"
        self.params = {}
        self.params[14] = (2.74, 14900, 2.15, 0.21)
        self.params[402] = (2.64, 600, 1.25, 0.14)
        self.params[1407] = (2.60, 33.2, 0.97, 0.01)  # CNO group, A = 14 (as 14N)
        self.params[2512] = (2.79, 34.2, 2.14, 0.01)  # Mg-Si group, A = 25 (as 25Mg)
        self.params[5626] = (2.68, 4.45, 3.07, 0.41)
        self.nucleus_ids = list(self.params.keys())

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)
        A = self.Z_A(corsika_id)[1]

        alpha = self.params[corsika_id][0]
        K = self.params[corsika_id][1]
        b = self.params[corsika_id][2]
        c = self.params[corsika_id][3]

        E = np.atleast_1d(np.asarray(E, dtype=float))
        E_k = E / A - NUCLEON_MASS_GEV
        flux = np.zeros_like(E)
        above = E_k > 0
        e_k = E_k[above]
        flux[above] = K / A * (e_k + b * np.exp(-c * np.sqrt(e_k))) ** -alpha
        return flux


class Thunman(PrimaryFlux):
    """Popular broken power-law flux model.

    The parameters of this model are taken from the prompt flux calculation
    paper by M. Thunman, G. Ingelman, and P. Gondolo,
    Astroparticle Physics 5, 309 (1996).
    The model contains only protons with a power-law index of -2.7 below
    the knee, located at 5 PeV, and -3.0 for energies higher than that.
    """

    name = "Thunman et al. ('96)"
    sname = "TIG"

    def __init__(self, *args, **kwargs):
        super().__init__(geomagnetic_cutoff=kwargs.pop("geomagnetic_cutoff", None))
        self.params = {}
        self.params["low_e"] = (1e4 * 1.7, -2.7)
        self.params["high_e"] = (1e4 * 174, -3.0)
        self.params["trans"] = 5e6

        self.nucleus_ids = [14]

    def _nucleus_flux(self, corsika_id, E):
        """Broken power law spectrum for protons."""
        E = np.atleast_1d(np.asarray(E, dtype=float))
        if corsika_id != 14:
            return np.zeros_like(E)

        le = E < self.params["trans"]
        he = ~le
        result = np.empty_like(E)
        result[le] = self.params["low_e"][0] * E[le] ** self.params["low_e"][1]
        result[he] = self.params["high_e"][0] * E[he] ** self.params["high_e"][1]
        return result


class SimplePowerlaw27(PrimaryFlux):
    """Simple E**-2.7 parametrization based on values from
    :class:`Thunman` below knee.
    """

    name = r"$E^{-2.7}$"
    sname = r"$E^{-2.7}$"

    def __init__(self, *args, **kwargs):
        super().__init__(geomagnetic_cutoff=kwargs.pop("geomagnetic_cutoff", None))
        self.params = (1e4 * 1.7, -2.7)
        self.nucleus_ids = [14]

    def _nucleus_flux(self, corsika_id, E):
        E = np.atleast_1d(np.asarray(E, dtype=float))
        if corsika_id != 14:
            return np.zeros_like(E)

        return self.params[0] * E ** (self.params[1])


def _gsf_options(version, time_interval, geomagnetic_cutoff, cutoff_width):
    """Keyword arguments for the globalsplinefit model constructors."""
    opts = {
        "version": version,
        "default_time_interval": time_interval,
        "default_rigidity_cutoff": geomagnetic_cutoff,
    }
    if cutoff_width is not None:
        opts["cutoff_width"] = cutoff_width
    return opts


class GlobalSplineFit(PrimaryFlux):
    """Global Spline Fit (GSF) model through the globalsplinefit package.

    Per-nucleus fluxes come from ``GSFEnergy`` (total energy per nucleus) and
    nucleon fluxes from ``GSFEnergyPerNucleon.p_and_n_flux`` (total energy per
    nucleon), both in (m² s sr GeV)^-1. Requires the optional ``gsf`` extra.

    Use for reference: Fedynitch et al., arXiv:2609.32649 (2026).

    Parameters
    ----------
    version : str
        globalsplinefit model version, e.g. "2026.1", "2025", "2019", "2017".
    time_interval : tuple or str, optional
        Solar-modulation interval (YYYYMM, YYYYMM), or "LIS". None uses the
        Solar Cycle 24 average.
    geomagnetic_cutoff : float, optional
        Rigidity cutoff in GV, applied by globalsplinefit per species in
        true rigidity. ``None`` or 0 means no cutoff.
    cutoff_width : float, optional
        Width in GV of globalsplinefit's sigmoid cutoff transition; 0 gives a
        sharp step at ``geomagnetic_cutoff``. ``None`` keeps the
        globalsplinefit default (1 GV).

    Notes
    -----
    ``nucleus_ids`` lists every species of the version as CORSIKA ids
    (100 A + Z with A the rounded mass, 14 for protons, 201 for deuterium
    where the version carries it). ``gsf`` is the underlying
    ``GSFEnergy`` model for uncertainties, covariances and composition.
    """

    def __init__(
        self,
        version="2026.1",
        time_interval=None,
        geomagnetic_cutoff=None,
        cutoff_width=None,
    ):
        super().__init__(geomagnetic_cutoff=geomagnetic_cutoff)
        from globalsplinefit import GSFEnergy, GSFEnergyPerNucleon

        opts = _gsf_options(version, time_interval, geomagnetic_cutoff, cutoff_width)
        self.gsf = GSFEnergy(**opts)
        self._gsf_nucleon = GSFEnergyPerNucleon(**opts)
        self._species = {self._corsika_id(*sid): sid for sid in self.gsf.species}
        self.nucleus_ids = sorted(self._species)
        self.name = f"Global Spline Fit {self.gsf.version}"
        self.sname = f"GSF{self.gsf.version}"

    @staticmethod
    def _corsika_id(z, a):
        a = int(round(a))
        return 14 if (z, a) == (1, 1) else 100 * a + z

    def nucleus_flux(self, corsika_id, E):
        # the cutoff is already applied by globalsplinefit
        return self._nucleus_flux(corsika_id, np.atleast_1d(np.asarray(E, dtype=float)))

    def _nucleus_flux(self, corsika_id, E):
        if corsika_id not in self._species:
            return np.zeros_like(E)
        return self.gsf.flux(E, self._species[corsika_id])

    def p_and_n_flux(self, E):
        energy = np.atleast_1d(np.asarray(E, dtype=float))
        model = self._gsf_nucleon
        proton, neutron = sum(
            model.p_and_n_flux(energy, g) for g in model.active_groups
        )
        total = proton + neutron
        fraction = np.divide(proton, total, out=np.zeros_like(total), where=total > 0)
        return fraction, proton, neutron

    def tot_nucleon_flux(self, E):
        return np.sum(self.p_and_n_flux(E)[1:], axis=0)


# PrimaryFlux is repeated as a direct base: MCEq accepts a primary-model object
# only if a direct base is named PrimaryFlux (initial_state.set_primary_model).
class GlobalSplineFitReduced(GlobalSplineFit, PrimaryFlux):
    """Reduced GSF proton/neutron model (``globalsplinefit.ReducedGSF``).

    The nucleon flux of :class:`GlobalSplineFit` with its uncertainty
    represented by relative-flux deviations ``theta`` at pivot energies;
    ``theta[i] = reduction.sigma[i]`` is a one-sigma variation. Nucleon fluxes
    only: ``nucleus_ids`` is empty and ``nucleus_flux``, ``total_flux`` and
    ``lnA`` raise ``NotImplementedError``.

    A ``geomagnetic_cutoff`` is applied by globalsplinefit to every species
    with its own (Z, A) before the reduction, so the central flux and the
    pivot covariance are those of the cut flux (the mask does not depend on
    the fit parameters). It is one global value per reduction: a cutoff that
    varies with direction needs one reduction per value, or
    :class:`GlobalSplineFit`. With a sharp step (``cutoff_width=0``) pivots
    below the cutoff have zero flux and globalsplinefit rejects the reduction.

    Use for reference: Fedynitch et al., arXiv:2609.32649 (2026).

    Parameters
    ----------
    version : str
        globalsplinefit model version; default "2026.1".
    theta : array-like, optional
        Reduced-model parameter vector; the default is the central flux.
    reduction : globalsplinefit.ReducedGSF, optional
        Reuse an existing total p/n reduction across many parameter variations
        (time interval, cutoff and basis are then taken from it).
    time_interval : tuple or str, optional
        Solar-modulation interval, or "LIS". None uses Solar Cycle 24.
    geomagnetic_cutoff : float, optional
        Rigidity cutoff in GV, applied by globalsplinefit. ``None`` or 0 means
        no cutoff.
    cutoff_width : float, optional
        Width in GV of the sigmoid cutoff transition (0 = sharp); ``None``
        keeps the globalsplinefit default (1 GV).
    basis : str
        Interpolation basis ("spline" or "hat") when building a reduction.
    """

    def __init__(
        self,
        version="2026.1",
        theta=None,
        *,
        reduction=None,
        time_interval=None,
        geomagnetic_cutoff=None,
        cutoff_width=None,
        basis="spline",
    ):
        PrimaryFlux.__init__(self, geomagnetic_cutoff=geomagnetic_cutoff)
        from globalsplinefit import (
            GSFEnergyPerNucleon,
            GSFKineticEnergyPerNucleon,
            ReducedGSF,
        )

        if reduction is None:
            reduction = ReducedGSF(
                GSFEnergyPerNucleon(
                    **_gsf_options(
                        version, time_interval, geomagnetic_cutoff, cutoff_width
                    )
                ),
                basis=basis,
            )
        elif (
            time_interval is not None
            or geomagnetic_cutoff is not None
            or cutoff_width is not None
            or basis != "spline"
        ):
            raise ValueError(
                "Configure time_interval/geomagnetic_cutoff/cutoff_width/basis "
                "on the supplied reduction"
            )
        if not isinstance(reduction, ReducedGSF) or reduction.per_group:
            raise ValueError("A total proton/neutron ReducedGSF is required")
        if not isinstance(reduction.model, GSFEnergyPerNucleon) or isinstance(
            reduction.model, GSFKineticEnergyPerNucleon
        ):
            raise ValueError(
                "MCEq requires total energy per nucleon, not kinetic energy"
            )
        self.reduction = reduction
        self.gsf = reduction.model
        self.nucleus_ids = []
        self.theta = (
            np.zeros(reduction.n_params)
            if theta is None
            else np.array(theta, dtype=float, copy=True)
        )
        if (
            self.theta.shape != (reduction.n_params,)
            or not np.isfinite(self.theta).all()
        ):
            raise ValueError(
                f"theta must contain {reduction.n_params} finite components"
            )
        self.theta.setflags(write=False)
        self.name = f"Global Spline Fit {reduction.model.version} reduced"
        self.sname = f"GSF{reduction.model.version}"

    def _nucleus_flux(self, corsika_id, E):
        raise NotImplementedError(_nucleon_only_message(self, "nucleus_flux"))

    def p_and_n_flux(self, E):
        energy = np.atleast_1d(np.asarray(E, dtype=float))
        proton, neutron = self.reduction.flux(energy, self.theta)
        total = proton + neutron
        fraction = np.divide(proton, total, out=np.zeros_like(total), where=total > 0)
        return fraction, proton, neutron


class GlobalSplineFitBeta(PrimaryFlux):
    """Data-driven fit of direct and indirect measurements of the
    cosmic ray flux and composition. Tracks the mass composition using
    four leading elements (p, He, O, Fe), whose flux is modeled by shaped
    spline functions. Assumes fixed flux ratios in rigidity for the flux
    of subleading elements. Covers the whole rigidity range from
    10 GV to 10^11 GeV.

    .. deprecated::
        Tabulated nucleon flux of GSF 2017 at Earth, modulated with a fixed
        force-field potential of 554 MV (Usoskin, July 1998). It equals
        globalsplinefit ``GSFEnergyPerNucleon(version="2017")`` with
        ``time_interval=(199807, 199808)`` (protons to 0.2%; neutrons differ by
        a few percent from the nucleon-count convention). Use
        ``GlobalSplineFit(version="2017")`` (or "2019", or a current version)
        instead. The class picks the most recent spline file in the package
        directory.

    .. note::
        The table holds nucleon fluxes without composition, so a rigidity
        cutoff cannot be applied (it needs each nucleus' Z and A):
        ``geomagnetic_cutoff`` other than ``None``/0 raises ``ValueError``,
        and ``nucleus_flux``, ``total_flux`` and ``lnA`` raise
        ``NotImplementedError``. ``GlobalSplineFit`` provides both.

    Use for reference: Dembinski et al., PoS ICRC2017 533
    https://inspirehep.net/literature/1639832
    """

    name = "Global Spline Fit"
    sname = "GSF"

    def __init__(self, spl_fname=None, **kwargs):
        warnings.warn(
            "GlobalSplineFitBeta is deprecated: it tabulates the GSF 2017 "
            "nucleon flux at phi = 554 MV. Use globalsplinefit "
            "crflux.models.GlobalSplineFit(version='2017') instead (or '2019', "
            "or a current version).",
            DeprecationWarning,
            stacklevel=2,
        )
        if kwargs.get("geomagnetic_cutoff"):
            raise ValueError(
                "GlobalSplineFitBeta has no composition, so a rigidity cutoff "
                "cannot be applied. Use crflux.models.GlobalSplineFit, the full "
                "per-species GSF model, with geomagnetic_cutoff."
            )
        super().__init__(**kwargs)
        import bz2
        import os
        import os.path as path
        import pickle

        base_path = path.dirname(path.abspath(__file__))

        if spl_fname is None:
            # Look for all files starting with GSF_spline
            gsf_files = [
                fname
                for fname in os.listdir(base_path)
                if fname.startswith("GSF_spline_")
            ]

            if not gsf_files:
                raise Exception(
                    self.__class__.__name__ + ": No spline files found in " + base_path
                )

            spl_fname = gsf_files[0]

            # Find out file datetag
            def fdate(fn):
                return int(os.path.splitext(os.path.splitext(fn)[0])[0].split("_")[-1])

            # Pick the latest
            for fn in gsf_files:
                if fdate(fn) >= fdate(spl_fname):
                    spl_fname = fn
            spl_fname = os.path.join(base_path, spl_fname)

        self.p_frac_spl, self.p_flux_spl, self.n_flux_spl = pickle.load(
            bz2.BZ2File(spl_fname), encoding="latin1"
        )

        self.nucleus_ids = []

    def p_and_n_flux(self, E):
        """Returns tuple with proton fraction, proton flux and neutron flux.

        The proton fraction is defined as
        :math:`\\frac{\\Phi_p}{\\Phi_p + \\Phi_n}`.

        Args:
          E (float or numpy.ndarray): laboratory energy of nucleons in GeV
        Returns:
          (float,float,float): proton fraction, proton flux, neutron flux
        """
        E = np.atleast_1d(np.asarray(E, dtype=float))
        p_frac = self.p_frac_spl(np.log(E))
        p_flux = np.exp(self.p_flux_spl(np.log(E)))
        n_flux = np.exp(self.n_flux_spl(np.log(E)))

        return p_frac, p_flux, n_flux

    def tot_nucleon_flux(self, E):
        """Returns total flux of nucleons, the "all-nucleon-flux".

        Args:
          E (float or numpy.ndarray): laboratory energy of nucleons in GeV
        Returns:
          (numpy.ndarray): nucleon flux :math:`\\Phi_{nucleons}` in
          :math:`(\\text{m}^2 \\text{s sr GeV})^{-1}`
        """
        return np.sum(self.p_and_n_flux(E)[1:], axis=0)

    def _nucleus_flux(self, corsika_id, E):
        """Not available: the table holds nucleon fluxes only."""
        raise NotImplementedError(_nucleon_only_message(self, "nucleus_flux"))
