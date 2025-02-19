import numpy as np
from astropy import units as u
#from ..util import sfh_check,unit_check

#Generate n random numbers from a uniform distribution over 0 to maxtime    
def make_random(self,n,max_val):
    rng = np.random.default_rng()
    return u.Quantity(rng.uniform(0,max_val.value,n),max_val.unit)

#Generate n random numbers from a normal distribution centered on 0 with a 1-sigma of interval                                                             
def make_normal_random(self,n,sigma):
    rng = np.random.default_rng()
    return u.Quantity(rng.normal(0,sigma.value,n),sigma.unit)

#Wrapper for generating random ages                                         
def make_offset(sfh,n_times,timescale):
    #sfh_check(sfh)
    #unit_check(timescale,'time')
    if sfh == 'random':
        return make_random(n_times,timescale)
    else:
        return make_normal_random(n_times,timescale)
