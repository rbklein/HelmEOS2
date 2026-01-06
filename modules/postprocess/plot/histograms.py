from config.conf_thermodynamics import *

from modules.thermodynamics.EOS import pressure
from numpy import linspace, ravel, log10, histogram2d

rho_c, T_c, p_c = molecule.critical_point

def compute_pT_histogram(u, T, p_range=(1.0, 2.0), T_range=(1.0, 2.0), n_bins=200):
    """
    Dimension-agnostic (1D/2D/3D) histogram over p and T.

    u shape: (nvar, ...), where ... is 1D/2D/3D mesh
    T shape: same as u[0]
    """
    rho = u[0]
    if rho.shape != T.shape:
        raise ValueError(f"Shape mismatch: rho.shape={rho.shape} vs T.shape={T.shape}")

    # compute pressure on the full mesh
    p = pressure(rho, T)

    # bin edges
    p_edges = linspace(p_range[0], p_range[1], n_bins + 1) * p_c
    T_edges = linspace(T_range[0], T_range[1], n_bins + 1) * T_c

    # flatten to 1D samples
    p1 = ravel(p)
    T1 = ravel(T)

    # 2D histogram: note order (x, y) => (p, T)
    H, _, _ = histogram2d(p1, T1, bins=(p_edges, T_edges))

    # centers + log scaling
    p_cent = 0.5 * (p_edges[:-1] + p_edges[1:])
    T_cent = 0.5 * (T_edges[:-1] + T_edges[1:])
    Z = log10(H + 1)

    return H, Z, p_cent, T_cent, p_edges, T_edges
