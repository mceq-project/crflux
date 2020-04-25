from matplotlib import pyplot as plt
import crflux.models as mods
import numpy as np

def test_plotting():
    
    pmodels = [
        (mods.GaisserStanevTilav, "3-gen", "GST 3-gen", "b", "--"),
        (mods.GaisserStanevTilav, "4-gen", "GST 4-gen", "b", "-"),
        (mods.CombinedGHandHG, "H3a", "cH3a", "g", "--"),
        (mods.CombinedGHandHG, "H4a", "cH4a", "g", "-"),
        (mods.HillasGaisser2012, "H3a", "H3a", "r", "--"),
        (mods.HillasGaisser2012, "H4a", "H4a", "r", "-"),
        (mods.PolyGonato, False, "poly-gonato", "m", "-"),
        (mods.Thunman, None, "TIG", "y", "-"),
        (mods.ZatsepinSokolskaya, 'default', 'ZS', "c", "-"),
        (mods.ZatsepinSokolskaya, 'pamela', 'ZSP', "c", "--"),
        (mods.GaisserHonda, None, 'GH', "0.5", "-"),
        #    (GlobalSplineFit, None, 'GSF', "k", "-"),
        (mods.GlobalSplineFitBeta, None, 'GSF spl', "k", ":")
    ]

    nfrac = {}
    lnA = {}
    evec = np.logspace(0, 11, 1000)
    plt.figure(figsize=(7.5, 5))
    plt.title('Cosmic ray nucleon flux (proton + neutron)')
    for mclass, moptions, mtitle, color, ls in pmodels:

        pmod = mclass(moptions)
        pfrac, p, n = pmod.p_and_n_flux(evec)
        plt.plot(
            evec, (p + n) * evec**2.5,
            color=color,
            ls=ls,
            lw=1.5,
            label=mtitle)
        nfrac[mtitle] = (1 - pfrac)
        if isinstance(pmod, mods.GlobalSplineFitBeta):
            continue
        lnA[mtitle] = pmod.lnA(evec)

    plt.loglog()
    plt.xlabel(r"$E_{nucleon}$ [GeV]")
    plt.ylabel(r"dN/dE (E/GeV)$^{2.5}$ (m$^{2}$ s sr GeV)$^{-1}$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e11])
    plt.ylim([10, 2e4])
    plt.tight_layout()


    plt.figure(figsize=(7.5, 5))
    plt.title('Fraction of neutrons relative to protons.')
    for mclass, moptions, mtitle, color, ls in pmodels:
        plt.plot(evec, nfrac[mtitle], color=color, ls=ls, lw=1.5, label=mtitle)

    plt.semilogx()
    plt.xlabel(r"$E_{nucleon}$ [GeV]")
    plt.ylabel("Neutron fraction")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e11])
    plt.tight_layout()
    
    pmodels = [m for m in pmodels if 'GSF' not in m[2]]
    plt.figure(figsize=(7.5, 5))
    plt.title('Cosmic ray particle flux (all-nuclei).')

    for mclass, moptions, mtitle, color, ls in pmodels:
        pmod = mclass(moptions)

        flux = pmod.total_flux(evec)
        plt.plot(
            evec, flux * evec**2.5, color=color, ls=ls, lw=1.5, label=mtitle)

    plt.loglog()
    plt.xlabel(r"$E_{particle}$ [GeV]")
    plt.ylabel(r"dN/dE (E/GeV)$^{2.5}$ (m$^{2}$ s sr GeV)$^{-1}$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e11])
    plt.ylim([10, 2e4])
    plt.tight_layout()

    plt.figure(figsize=(7.5, 5))
    plt.title('Mean log mass <lnA>.')
    for mclass, moptions, mtitle, color, ls in pmodels:
        plt.plot(evec, lnA[mtitle], color=color, ls=ls, lw=1.5, label=mtitle)

    plt.semilogx()
    plt.xlabel(r"$E_{particle}$ [GeV]")
    plt.ylabel(r"$<\ln{A}>$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e11])
    plt.tight_layout()

    # plt.show()

