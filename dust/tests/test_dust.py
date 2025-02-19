import numpy as np
from astropy import units as u
from astropy.modeling.models import BlackBody

from dust_sphere import *

M_cl = 1000 * u.M_sun
efficiency = 33
wav = np.geomspace(1e-2,5e3,200) * u.um
aps = np.logspace(2,6,20) * u.AU
distance = 1 * u.kpc

T = 10 * u.K
R_cl = 1 * u.pc
mu = 2.4 * u.Da

bb = BlackBody(T)
kappa = get_kappa(wav) * u.cm**2 / u.g

initial_d = 1 * u.kpc
r_max = r_BE_max(M_cl,efficiency,T=T,R_cl=R_cl,mu=mu)
ang_size = (r_max**2 / 4 / initial_d**2).decompose().value * u.sr

ret = []
for ap in aps:
    sigma = avg_surf_dens(ap,M_cl,efficiency,T=T,R_cl=R_cl,mu=mu)
    S = (0.5 * bb(wav) * (1 - np.exp(-sigma * kappa)) * ang_size).to(u.mJy)
    S *= min((ap / r_max)**2,1)
    ret.append(scale_distance(S,distance))
