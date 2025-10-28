"""
Shared plotting functions for documentation and tests.

This module contains the plotting code used both in the module docstrings
(via Sphinx's plot directive) and in test_plotting.py to ensure consistency.
"""

import numpy as np
from matplotlib import pyplot as plt
import warnings


def generate_model_comparison_plots(save_dir=None):
    """
    Generate comparison plots for all cosmic ray flux models.
    
    This function creates four main plots:
    1. Nucleon flux (proton + neutron)
    2. Neutron fraction
    3. Total particle flux (all-nuclei)
    4. Mean log mass <lnA>
    
    If save_dir is provided, plots are saved as PNG files. Otherwise, they are displayed.
    
    Args:
        save_dir (str, optional): Directory path to save plots. If None, plots are shown.
    
    Returns:
        dict: Dictionary containing the plot data for further analysis
    """
    import crflux.models as mods
    
    # Suppress numpy deprecation warnings from the interpolation code
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='scipy')
    
    # Define all models to plot
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
        (mods.GlobalSplineFitBeta, None, 'GSF spl', "k", ":"),
    ]
    
    # Try to add GlobalSplineFit2025 variants if available
    gsf2025_available = False
    try:
        # Test that the model can be instantiated
        _ = mods.GlobalSplineFit2025(version="2025")
        pmodels.append((lambda opts: mods.GlobalSplineFit2025(version="2025"), None, 'GSF2025', "k", "-"))
        gsf2025_available = True
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        warnings.warn(f"GlobalSplineFit2025 (2025) not available: {e}", ImportWarning)
    
    try:
        _ = mods.GlobalSplineFit2025(version="2019")
        pmodels.append((lambda opts: mods.GlobalSplineFit2025(version="2019"), None, 'GSF2019', "k", "-."))
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        warnings.warn(f"GlobalSplineFit2025 (2019) not available: {e}", ImportWarning)
    
    try:
        _ = mods.GlobalSplineFit2025(version="2017")
        pmodels.append((lambda opts: mods.GlobalSplineFit2025(version="2017"), None, 'GSF2017', "k", "--"))
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        warnings.warn(f"GlobalSplineFit2025 (2017) not available: {e}", ImportWarning)
    
    # Calculate fluxes
    nfrac = {}
    lnA = {}
    nucleon_flux = {}
    evec = np.logspace(0, 11, 1000)
    
    # Plot 1: Nucleon flux
    plt.figure(figsize=(7.5, 5))
    plt.title('Cosmic ray nucleon flux (proton + neutron)')
    for mclass, moptions, mtitle, color, ls in pmodels:
        pmod = mclass(moptions)
        pfrac, p, n = pmod.p_and_n_flux(evec)
        nucleon_flux[mtitle] = p + n
        plt.plot(
            evec, (p + n) * evec**2.5,
            color=color,
            ls=ls,
            lw=1.5,
            label=mtitle)
        nfrac[mtitle] = (1 - pfrac)
        # Skip lnA for GlobalSplineFitBeta as it doesn't support it
        # but GlobalSplineFit2025 does support lnA
        if isinstance(pmod, mods.GlobalSplineFitBeta):
            continue
        try:
            lnA[mtitle] = pmod.lnA(evec)
        except (AttributeError, NotImplementedError):
            pass  # Skip models without lnA method
    
    plt.loglog()
    plt.xlabel(r"$E_{nucleon}$ [GeV]")
    plt.ylabel(r"dN/dE (E/GeV)$^{2.5}$ (m$^{2}$ s sr GeV)$^{-1}$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e11])
    plt.ylim([10, 2e4])
    plt.tight_layout()
    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, 'nucleon_flux.png'), dpi=150)
        plt.close()
    
    # Plot 2: Neutron fraction
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
    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, 'neutron_fraction.png'), dpi=150)
        plt.close()
    
    # Filter out GlobalSplineFitBeta for total flux and lnA plots
    # (GlobalSplineFit2025 supports these methods)
    pmodels_filtered = [m for m in pmodels if 'GSF spl' not in m[2]]
    
    # Plot 3: Total particle flux
    total_flux = {}
    plt.figure(figsize=(7.5, 5))
    plt.title('Cosmic ray particle flux (all-nuclei).')
    
    for mclass, moptions, mtitle, color, ls in pmodels_filtered:
        pmod = mclass(moptions)
        flux = pmod.total_flux(evec)
        total_flux[mtitle] = flux
        plt.plot(
            evec, flux * evec**2.5, color=color, ls=ls, lw=1.5, label=mtitle)
    
    plt.loglog()
    plt.xlabel(r"$E_{particle}$ [GeV]")
    plt.ylabel(r"dN/dE (E/GeV)$^{2.5}$ (m$^{2}$ s sr GeV)$^{-1}$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e11])
    plt.ylim([10, 2e4])
    plt.tight_layout()
    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, 'total_flux.png'), dpi=150)
        plt.close()
    
    # Plot 4: Mean log mass
    plt.figure(figsize=(7.5, 5))
    plt.title('Mean log mass <lnA>.')
    for mclass, moptions, mtitle, color, ls in pmodels_filtered:
        if mtitle in lnA:  # Only plot models that have lnA data
            plt.plot(evec, lnA[mtitle], color=color, ls=ls, lw=1.5, label=mtitle)
    
    plt.semilogx()
    plt.xlabel(r"$E_{particle}$ [GeV]")
    plt.ylabel(r"$<\ln{A}>$")
    plt.legend(loc=0, frameon=False, numpoints=1, ncol=2)
    plt.xlim([1, 1e11])
    plt.tight_layout()
    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, 'mean_log_mass.png'), dpi=150)
        plt.close()
    
    # Optional ratio plots if GSF2025 is available
    if gsf2025_available and 'GSF2025' in nucleon_flux:
        try:
            # Ratio plot for nucleon flux
            evec_ratio = np.logspace(0, 7, 500)  # Limited to 10^7 GeV
            plt.figure(figsize=(7.5, 5))
            plt.title('Nucleon flux ratio relative to GSF2025')
            
            # Get GSF2025 reference flux at ratio energy points (explicitly use version="2025")
            gsf2025_mod = mods.GlobalSplineFit2025(version="2025")
            _, p_ref, n_ref = gsf2025_mod.p_and_n_flux(evec_ratio)
            nucleon_flux_ref = p_ref + n_ref
            
            for mclass, moptions, mtitle, color, ls in pmodels:
                if mtitle == 'GSF2025':
                    continue  # Skip the reference itself
                
                pmod = mclass(moptions)
                pfrac, p, n = pmod.p_and_n_flux(evec_ratio)
                flux_ratio = (p + n) / nucleon_flux_ref
                
                plt.plot(evec_ratio, flux_ratio, color=color, ls=ls, lw=1.5, label=mtitle)
            
            plt.axhline(y=1.0, color='k', ls='-', lw=1.5, label='GSF2025', alpha=0.5)
            plt.semilogx()
            plt.xlabel(r"$E_{nucleon}$ [GeV]")
            plt.ylabel("Flux ratio relative to GSF2025")
            plt.legend(loc='best', frameon=False, numpoints=1, ncol=2, fontsize=8)
            plt.xlim([1, 1e7])
            plt.ylim([0.7, 1.3])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir:
                import os
                plt.savefig(os.path.join(save_dir, 'nucleon_flux_ratio.png'), dpi=150)
                plt.close()
            
            # Ratio plot for total nucleus flux
            if 'GSF2025' in total_flux:
                plt.figure(figsize=(7.5, 5))
                plt.title('Total particle flux ratio relative to GSF2025')
                
                # Get GSF2025 reference total flux at ratio energy points
                total_flux_ref = gsf2025_mod.total_flux(evec_ratio)
                
                for mclass, moptions, mtitle, color, ls in pmodels_filtered:
                    if mtitle == 'GSF2025':
                        continue  # Skip the reference itself
                    
                    pmod = mclass(moptions)
                    flux = pmod.total_flux(evec_ratio)
                    flux_ratio = flux / total_flux_ref
                    
                    plt.plot(evec_ratio, flux_ratio, color=color, ls=ls, lw=1.5, label=mtitle)
                
                plt.axhline(y=1.0, color='k', ls='-', lw=1.5, label='GSF2025', alpha=0.5)
                plt.semilogx()
                plt.xlabel(r"$E_{particle}$ [GeV]")
                plt.ylabel("Flux ratio relative to GSF2025")
                plt.legend(loc='best', frameon=False, numpoints=1, ncol=2, fontsize=8)
                plt.xlim([1, 1e7])
                plt.ylim([0.7, 1.3])
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                if save_dir:
                    import os
                    plt.savefig(os.path.join(save_dir, 'total_flux_ratio.png'), dpi=150)
                    plt.close()
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            warnings.warn(f"Could not generate GSF2025 ratio plots: {e}", ImportWarning)
    
    if not save_dir:
        plt.show()
    
    return {
        'nfrac': nfrac,
        'lnA': lnA,
        'nucleon_flux': nucleon_flux,
        'total_flux': total_flux,
        'evec': evec
    }


def test():
    """
    Test function to generate and display plots.
    Called when running: from crflux.models import test; test()
    """
    generate_model_comparison_plots(save_dir=None)
