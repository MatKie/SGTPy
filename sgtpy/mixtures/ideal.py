import numpy as np


h = 6.626070150e-34  # J s
me = 9.10938291e-31  # 1/Kg


# Ideal contribution
# Eq 68
def aideal(x, rho, beta):
    broglie_vol = h / np.sqrt(2*np.pi * me / beta)
    a = np.nan_to_num(np.dot(x, np.log(x)))
    a += np.log(rho * broglie_vol**3)
    a -= 1.
    return a


def daideal_drho(x, rho, beta):
    broglie_vol = h / np.sqrt(2*np.pi * me / beta)
    a = np.nan_to_num(np.dot(x, np.log(x)))
    a += np.log(rho * broglie_vol**3) - 1.
    da = 1./rho
    return np.hstack([a, da])


def d2aideal_drho(x, rho, beta):
    broglie_vol = h / np.sqrt(2*np.pi * me / beta)
    a = np.nan_to_num(np.dot(x, np.log(x)))
    a += np.log(rho * broglie_vol**3) - 1
    da = 1./rho
    d2a = -1/rho**2
    return np.hstack([a, da, d2a])


def daideal_dx(x, rho, beta):
    logx = np.log(x)
    broglie_vol = h / np.sqrt(2*np.pi * me / beta)
    a = np.nan_to_num(np.dot(x, logx))
    a += np.log(rho * broglie_vol**3)
    a -= 1.

    da = np.nan_to_num(logx) + 1.
    return a, da


def daideal_dxrho(x, rho, beta):
    logx = np.log(x)
    broglie_vol = h / np.sqrt(2*np.pi * me / beta)
    a = np.nan_to_num(np.dot(x, logx))
    a += np.log(rho * broglie_vol**3)
    a -= 1.
    da = 1./rho

    dax = np.nan_to_num(logx) + 1.
    return np.hstack([a, da]), dax
