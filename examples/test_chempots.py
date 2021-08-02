from SGTPy import component, mixture, saftvrmie
import numpy as np
from mkutils import create_fig, save_to_file 
from scipy import integrate

# creating pure components - same water than in gustavos examples
# Association scheme?
water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
                    lambda_r = 8.308, lambda_a = 6., eAB = 1991.07, rcAB = 0.5624,
                    rdAB = 0.4, sites = [0,2,2], cii = 1.5371939421515458e-20)

octane = component('octane', ms = 3., sigma = 4.227 , eps = 333.7,
                    lambda_r = 16.14, lambda_a = 6.)
mix = mixture(water, octane)
eos_water = saftvrmie(water)
eos_octane = saftvrmie(octane)

T = 298.15
P = 101325

z = (-3.5450933661557516e-06, 2.6819855363061289e-03, -4.3239556892743974e-01)
p = np.poly1d(z)
# To be optimized from experimental LLE
kij = p(T)

Kij = np.array([[0., kij], [kij, 0.]])

# setting interactions corrections
mix.kij_saft(Kij)
# creating eos model
eos = saftvrmie(mix)
eos_pure = saftvrmie(octane)

# +
def get_lngamma(xi, T, p, eos, eos_pure=[None, None]):
    bool_eos = [True for eos_i in eos_pure if eos_i is not None]
    if not all(bool_eos):
        raise ValueError('Supply at least one pure eos object')

    if isinstance(xi, (float, int)):
        xi = np.array([xi, 1.-xi])
    elif not isinstance(xi, np.ndarray):
        xi = np.array(xi)
    
    if eos.nc > 2 and xi.shape[0] < 2:
        raise ValueError('Please supply the whole molfrac vector for non-binary mixtures')

    lnphi_mix, _ = eos.logfugef(xi, T, p, 'L')
    lnphi_pure = np.zeros_like(xi)
    for i, eos_pure_i in enumerate(eos_pure):
        if eos_pure_i is not None:
            lnphi_pure[i], _ = eos_pure_i.logfug(T, p, 'L')
        else:
            lnphi_pure[i] = np.nan    

    lngamma = lnphi_mix - lnphi_pure
    return lngamma


# -


x = np.logspace(-8, 0, 50)
x = np.ones_like(x) - x
y = [-get_lngamma(xi, T, P, eos, eos_pure=(eos_water, eos_octane))[-1] for xi in x]

fig, ax = create_fig(1,1)
ax = ax[0]
ax.plot(x,y, color='k', lw=2)
ax.set_xscale('logit')
ax.set_ylabel(r'$-\ln\left(\gamma_{\mathrm{Octane}}^W\right)$')
ax.set_xlabel(r'$x_{\mathrm{Octane}}^W$')

get_lngamma(0.1, T, P, eos, eos_pure=(eos_water, eos_octane)).shape

x.shape


