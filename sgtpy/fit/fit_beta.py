from __future__ import division, print_function, absolute_import
import warnings
import numpy as np
from ..sgt import sgt_mix
from scipy.optimize import minimize_scalar
from ..saft import saftvrmie


def fobj_beta(
    beta, iftexp, rho1, rho2, T, P, eos, rho0="linear", mixture=None, poly_kij=None
):
    if poly_kij is not None and mixture is not None:
        p = np.poly1d(poly_kij)
    else:
        bij = np.array([[0, beta], [beta, 0]])
        eos.beta_sgt(bij)
    tenb = np.zeros_like(iftexp)
    n = len(iftexp)

    n1, n2 = rho1.shape
    if n2 == n:
        rho1 = rho1.T
        rho2 = rho2.T

    for i in range(n):
        if poly_kij is not None and mixture is not None:
            kij = p(T[i])
            kij_mat = np.array([[0.0, kij], [kij, 0.0]])
            mixture.kij_saft(kij_mat)
            eos = saftvrmie(mixture)
            bij = np.array([[0, beta], [beta, 0]])
            eos.beta_sgt(bij)
        n = 20
        loop_bool = True
        # Increase number of colocation points twice by 5 to increase poss.
        # to converge.
        while loop_bool:
            sol = sgt_mix(
                rho1[i], rho2[i], T[i], P[i], eos, rho0=rho0, n=n, full_output=True
            )
            n += 5
            rho0 = sol
            loop_bool = not sol.success or n < 51

        if not sol.success:
            warnings.warn(
                "Could not converge point at temperature of {:.2f}K and beta{:.4f}. Error is {:.2e} and function norm {:.2e}".format(
                    T[i], beta, sol.error, sol.fun_norm
                )
            )

        tenb[i] = sol.tension
    fo = np.mean((1 - tenb / iftexp) ** 2)

    print(f"{tenb = }")
    print(f"{iftexp = }")
    print(f"{fo = }")
    print(f"{beta = }")
    return fo


def fit_beta(
    beta0, ExpTension, EquilibriumInfo, eos, rho0="linear", mixture=None, poly_kij=None
):
    """
    fit_beta
    Optimize beta for SGT for binary mixtures

    Parameters
    ----------
    beta0 : tuple
        boundaries for beta as needed for SciPy's minimize_scalar
    ExpTension : array
        Experimental interfacial tension of the mixture
    EquilibriumInfo : tuple
        tuple containing density vectors, temperature and pressure
        tuple = (rho1, rho2, T, P)
    eos : model
        saft vr mie model set up with the binary mixture

    Returns
    -------
    ten : OptimizeResult
        Result of SciPy minimize_scalar
    """
    rho1, rho2, T, P = EquilibriumInfo
    args = (ExpTension, rho1, rho2, T, P, eos, rho0, mixture, poly_kij)
    opti = minimize_scalar(fobj_beta, beta0, args=args)
    return opti
