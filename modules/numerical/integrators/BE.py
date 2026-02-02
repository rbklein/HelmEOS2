''' revise some day
from prep_jax import *

from modules.numerical.flux import dudt as dudt_c
from modules.numerical.viscous import dudt as dudt_v
from modules.numerical.heat import dudt as dudt_q
from modules.numerical.source import dudt as dudt_s
from modules.thermodynamics.EOS import temperature



from jax.numpy import asarray, inf
from jax.scipy.sparse.linalg import gmres


def _rhs(u, T, t):
    """Combined RHS: convective + viscous + heat."""
    return (
        dudt_c(u, T)
        + dudt_v(u, T)
        + dudt_q(u, T)
        + dudt_s(u, T, t)
    )


def backward_euler(u, T, dt, t):
    """
    One backward Euler step: solve
        u_new - u_old - dt * rhs(u_new) = 0
    using Newton–GMRES.

    Parameters
    ----------
    u : array-like
        Current state (u_old).
    T : array-like
        Temperature estimate used as initial guess for EOS.
    dt : float
        Time step size.

    Returns
    -------
    u_new : array-like
        Updated state after one implicit step.
    """
    u = jnp.asarray(u)
    T_guess = jnp.asarray(T)
    u_shape = u.shape

    # Flattened previous state (u_old), kept constant in the root equation
    u_prev_flat = u.reshape(-1)
    dt_arr = jnp.asarray(dt, dtype=u.dtype)

    # Root function F(v) = v - u_prev - dt * rhs(v)
    def root(v_flat):
        u_new = v_flat.reshape(u_shape)
        T_new = temperature(u_new, T_guess)
        rhs = _rhs(u_new, T_new, t + dt).reshape(-1)
        return v_flat - u_prev_flat - dt_arr * rhs

    # Initial guess for Newton: start from previous state
    v0_flat = u_prev_flat
    f0 = root(v0_flat)
    res0 = jnp.linalg.norm(f0, ord=jnp.inf)

    # Newton loop state: (k, v_flat, f_val, res_norm, gmres_info)
    init_state = (
        jnp.array(0, dtype=jnp.int32),   # iteration counter
        v0_flat,                         # current guess
        f0,                              # current residual
        res0,                            # current residual norm
        jnp.array(0, dtype=jnp.int32),   # last gmres info
    )

    newton_tol = 1e-8
    newton_maxiter = 20
    lin_tol = 1e-5
    lin_maxiter = 200

    def cond_fun(state):
        k, v_flat, f_val, res_norm, gmres_info = state
        return jnp.logical_and(res_norm > newton_tol, k < newton_maxiter)

    def body_fun(state):
        k, v_flat, f_val, res_norm, gmres_info = state

        # Linear operator: J(v) @ dx via jvp
        def matvec(dx):
            _, jv = jax.jvp(root, (v_flat,), (dx,))
            return jv

        # Solve J(v) * du = -F(v)
        du, info = gmres(
            matvec,
            -f_val,
            tol=lin_tol,
            atol=0.0,
            maxiter=lin_maxiter,
        )

        v_new = v_flat + du
        f_new = root(v_new)
        res_new = jnp.linalg.norm(f_new, ord=jnp.inf)

        return (
            k + 1,
            v_new,
            f_new,
            res_new,
            info.astype(jnp.int32),
        )

    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    _, v_final, _, _, _ = final_state

    u_new = v_final.reshape(u_shape)
    return u_new
'''
