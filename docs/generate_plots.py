#!/usr/bin/env python
"""Generate documentation plots for crflux.

Run from the repository root:
    python docs/generate_plots.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import crflux.models as mods

OUTDIR = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(OUTDIR, exist_ok=True)

PMODELS = [
    (mods.GaisserStanevTilav, "3-gen", "GST 3-gen", "#1f77b4", "--"),
    (mods.GaisserStanevTilav, "4-gen", "GST 4-gen", "#1f77b4", "-"),
    (mods.CombinedGHandHG, "H3a", "cH3a", "#2ca02c", "--"),
    (mods.CombinedGHandHG, "H4a", "cH4a", "#2ca02c", "-"),
    (mods.HillasGaisser2012, "H3a", "H3a", "#d62728", "--"),
    (mods.HillasGaisser2012, "H4a", "H4a", "#d62728", "-"),
    (mods.PolyGonato, False, "poly-gonato", "#9467bd", "-"),
    (mods.Thunman, None, "TIG", "#bcbd22", "-"),
    (mods.ZatsepinSokolskaya, "default", "ZS", "#17becf", "-"),
    (mods.ZatsepinSokolskaya, "pamela", "ZSP", "#17becf", "--"),
    (mods.GaisserHonda, None, "GH", "#7f7f7f", "-"),
    (mods.GlobalSplineFitBeta, None, "GSF spl", "k", ":"),
]

evec = np.logspace(0, 11, 1000)


def plot_nucleon_flux():
    fig, ax = plt.subplots(figsize=(8, 5))
    for mclass, moptions, mtitle, color, ls in PMODELS:
        pmod = mclass(moptions)
        _pfrac, p, n = pmod.p_and_n_flux(evec)
        ax.plot(
            evec,
            (p + n) * evec**2.5,
            color=color,
            ls=ls,
            lw=1.5,
            label=mtitle,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_\mathrm{nucleon}$ [GeV]")
    ax.set_ylabel(
        r"$dN/dE \cdot (E/\mathrm{GeV})^{2.5}$"
        r" $(\mathrm{m}^{2}\,\mathrm{s}\,\mathrm{sr}\,\mathrm{GeV})^{-1}$"
    )
    ax.set_xlim(1, 1e11)
    ax.set_ylim(10, 2e4)
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=8)
    ax.set_title("Cosmic ray nucleon flux (proton + neutron)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "nucleon_flux.png"), dpi=150)
    plt.close(fig)


def plot_neutron_fraction():
    nfrac = {}
    for mclass, moptions, mtitle, color, ls in PMODELS:
        pmod = mclass(moptions)
        pfrac, _p, _n = pmod.p_and_n_flux(evec)
        nfrac[mtitle] = 1 - pfrac

    fig, ax = plt.subplots(figsize=(8, 5))
    for _mclass, _moptions, mtitle, color, ls in PMODELS:
        ax.plot(evec, nfrac[mtitle], color=color, ls=ls, lw=1.5, label=mtitle)
    ax.set_xscale("log")
    ax.set_xlabel(r"$E_\mathrm{nucleon}$ [GeV]")
    ax.set_ylabel("Neutron fraction")
    ax.set_xlim(1, 1e11)
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=8)
    ax.set_title("Fraction of neutrons relative to protons")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "neutron_fraction.png"), dpi=150)
    plt.close(fig)


def plot_total_flux():
    # Exclude spline-only models that don't support total_flux
    pmodels_nuc = [m for m in PMODELS if "GSF" not in m[2]]

    fig, ax = plt.subplots(figsize=(8, 5))
    for mclass, moptions, mtitle, color, ls in pmodels_nuc:
        pmod = mclass(moptions)
        flux = pmod.total_flux(evec)
        ax.plot(
            evec,
            flux * evec**2.5,
            color=color,
            ls=ls,
            lw=1.5,
            label=mtitle,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_\mathrm{particle}$ [GeV]")
    ax.set_ylabel(
        r"$dN/dE \cdot (E/\mathrm{GeV})^{2.5}$"
        r" $(\mathrm{m}^{2}\,\mathrm{s}\,\mathrm{sr}\,\mathrm{GeV})^{-1}$"
    )
    ax.set_xlim(1, 1e11)
    ax.set_ylim(10, 2e4)
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=8)
    ax.set_title("Cosmic ray particle flux (all-nuclei)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "total_flux.png"), dpi=150)
    plt.close(fig)


def plot_lnA():
    pmodels_nuc = [m for m in PMODELS if "GSF" not in m[2]]
    lnA = {}
    for mclass, moptions, mtitle, color, ls in pmodels_nuc:
        pmod = mclass(moptions)
        lnA[mtitle] = pmod.lnA(evec)

    fig, ax = plt.subplots(figsize=(8, 5))
    for _mclass, _moptions, mtitle, color, ls in pmodels_nuc:
        ax.plot(evec, lnA[mtitle], color=color, ls=ls, lw=1.5, label=mtitle)
    ax.set_xscale("log")
    ax.set_xlabel(r"$E_\mathrm{particle}$ [GeV]")
    ax.set_ylabel(r"$\langle\ln A\rangle$")
    ax.set_xlim(1, 1e11)
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=8)
    ax.set_title(r"Mean log mass $\langle\ln A\rangle$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "lnA.png"), dpi=150)
    plt.close(fig)


def plot_geomagnetic_cutoff():
    fig, ax = plt.subplots(figsize=(8, 5))
    E = np.logspace(0, 7, 1000)

    mod = mods.HillasGaisser2012("H3a")
    _pfrac, p, n = mod.p_and_n_flux(E)
    ax.plot(
        E, (p + n) * E**2.5, color="#1f77b4", lw=1.5, label="H3a (no cutoff)"
    )

    for rcut, color, ls in [
        (5.0, "#d62728", "--"),
        (10.0, "#2ca02c", "-."),
        (15.0, "#9467bd", ":"),
    ]:
        mod_cut = mods.HillasGaisser2012("H3a", geomagnetic_cutoff=rcut)
        _pfrac, p, n = mod_cut.p_and_n_flux(E)
        ax.plot(
            E,
            (p + n) * E**2.5,
            color=color,
            ls=ls,
            lw=1.5,
            label=f"H3a, $R_\\mathrm{{cut}}$ = {rcut:.0f} GV",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_\mathrm{nucleon}$ [GeV]")
    ax.set_ylabel(
        r"$dN/dE \cdot (E/\mathrm{GeV})^{2.5}$"
        r" $(\mathrm{m}^{2}\,\mathrm{s}\,\mathrm{sr}\,\mathrm{GeV})^{-1}$"
    )
    ax.set_xlim(1, 1e7)
    ax.set_ylim(10, 2e4)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.set_title("Effect of geomagnetic cutoff on nucleon flux")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "geomagnetic_cutoff.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_nucleon_flux()
    print("  nucleon_flux.png")
    plot_neutron_fraction()
    print("  neutron_fraction.png")
    plot_total_flux()
    print("  total_flux.png")
    plot_lnA()
    print("  lnA.png")
    plot_geomagnetic_cutoff()
    print("  geomagnetic_cutoff.png")
    print("Done.")
