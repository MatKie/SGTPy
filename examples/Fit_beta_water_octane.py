#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

from SGTPy import component, mixture, saftvrmie
from SGTPy.equilibrium import bubblePy, lle, lle_init
from SGTPy.fit import fit_beta


# In[15]:


water = component('water', ms = 1.7311, sigma = 2.4539 , eps = 110.85,
                    lambda_r = 8.308, lambda_a = 6., eAB = 1991.07, rcAB = 0.5624,
                    rdAB = 0.4, sites = [0,2,2], cii = 1.5371939421515458e-20)
                    
octane = component('octane', ms = 3., sigma = 4.227 , eps = 333.7,
                    lambda_r = 16.14, lambda_a = 6., cii=5.8915398756858572e-19)

mix = mixture(octane, water)
poly_kij = (-3.5450933661557516e-06,	2.6819855363061289e-03,	-4.3239556892743974e-01)
poly = np.poly1d(poly_kij)

# Experimental Data
Texp = np.array([293.15, 298.15, 308.65, 323.15])
Pexp = np.array([101325]*Texp.shape[0])
tension_exp = np.array([51.64, 51.16, 50.22, 48.95])

v_ar = np.zeros_like(Texp)
v_wr = np.zeros_like(Texp)

X_wr = np.array([1e-8, 1-1e-8])
X_ar = np.array([1.-1.e-4, 1.e-4])
Xfin_wr = np.zeros((X_wr.shape[0], Texp.shape[0]))
Xfin_ar = np.zeros((X_wr.shape[0], Texp.shape[0]))

for i, (T, p) in enumerate(zip(Texp, Pexp)):
    kij = poly(T)
    Kij = np.array([[0, kij], [kij, 0]])
    mix.kij_saft(Kij)
    eos = saftvrmie(mix)
    X_wr, X_ar = lle_init((X_wr+X_ar)/2., T, p, eos)
    sol = lle(X_wr, X_ar, (X_wr+X_ar)/2., T, p, eos, full_output = True)
    X_wr, X_ar = sol.X
    Xfin_wr[:, i] = X_wr
    Xfin_ar[:, i] = X_ar
    P = sol.P
    vl1, vl2 = sol.v

    #computing the density vector
    rhol_wr = X_wr / vl1
    rhol_ar = X_ar / vl2
    v_wr[i] = vl1
    v_ar[i] = vl2
    print(f'{X_wr = }')
    print(f'{X_ar = }')
    print(f'{vl1 = }')
    print(f'{vl2 = }')
    print(f'{rhol_wr = }')
    print(f'{rhol_ar = }')
rho_ar = Xfin_ar/v_ar
rho_wr = Xfin_wr/v_wr

print(f'{rho_ar = }')
print(f'{rho_wr = }')
print(f'{Xfin_wr = }')
print(f'{Xfin_ar = }')

# Beta optimization can be slow
EquilibriumInfo = (rho_wr, rho_ar, Texp, Pexp)
beta_bounds = (1e-6, 1.0-1e-6)
out = fit_beta(beta_bounds, tension_exp, EquilibriumInfo, eos, mixture=mix, poly_kij=poly_kij)

print(out)
try:
    print('{:.16e}'.format(out.x))
except:
    print('{:.16e}'.format(out.x[0]))

# For more information just run:
# ```fit_beta?```

# In[ ]:



