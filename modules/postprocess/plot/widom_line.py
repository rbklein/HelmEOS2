"""
Compute the widom line in a bounding box

Important: this approach is not jit-compilable at the moment
"""

from prep_jax import *
from modules.thermodynamics.EOS import *
from modules.postprocess.plot.isobar import isobar_T
from modules.numerical.computation import extract_1d_from_padded, pad_1d_to_mesh


def discrete_max_index(x: jnp.ndarray):
    # Same idea as your function, but returns a Python int or None
    # (so your `if i_max != None:` logic keeps working).
    n = x.shape[0]
    if n < 3:
        return None

    interior = x[1:-1]
    left = x[:-2]
    right = x[2:]

    mask = (interior > left) & (interior > right)
    exists = bool(jnp.any(mask))
    if not exists:
        return None

    idx0 = int(jnp.argmax(mask.astype(jnp.int32)))
    return idx0 + 1


def widom_line(p1, p2, T1, T2, num = 10):
    """
    Compute the widom line between in the box [T1, T2] x [p1, p2]
    """

    n_points_per_pass = 11   
    n_refine_passes = 4      

    ps = jnp.linspace(p1, p2, num)
    Ts = []
    rhos = []

    C_p = jax.jit(c_p)

    for p in ps:
        T_low, T_high = T1, T2

        T_peak = None
        rho_peak = None

        for _ in range(n_refine_passes):
            rho, T = isobar_T(p, T_low, T_high, n_points_per_pass)
            cp = extract_1d_from_padded(
                C_p(pad_1d_to_mesh(rho), pad_1d_to_mesh(T))
            )

            i_max = discrete_max_index(cp)
            if i_max is None:
                T_peak = None
                rho_peak = None
                break

            # after i_max is found
            T_peak = T[i_max]
            rho_peak = rho[i_max]

            # new bounds from neighboring samples (with safety)
            if i_max > 0 and i_max < (T.shape[0] - 1):
                T_low, T_high = T[i_max - 1], T[i_max + 1]
            else:
                break

        if T_peak is not None:
            Ts.append(T_peak)
            rhos.append(rho_peak)
        else:
            Ts.append(jnp.nan)
            rhos.append(jnp.nan)

    Ts = jnp.array(Ts)
    rhos = jnp.array(rhos)

    return rhos, Ts, ps