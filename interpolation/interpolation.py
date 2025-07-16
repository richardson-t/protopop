import numpy as np
from astropy.table import Table,vstack
from astropy import units as u
from astropy.modeling.models import BlackBody

#from ..util import interp_props

# find rows from the beginning
def find_row(selected_time, tbl):
    times = tbl['Time']
    selected_row = np.argmin(np.abs(times - selected_time))    
    return selected_row

def make_bb(temp,rad,
            distance,wav,aps):
    bb = BlackBody(temp)
    s_nu = (bb(wav) * rad**2 / 4 / distance**2 * u.sr).decompose().to(u.mJy) 
    ret = []
    for inc in range(9):
        ret.append([s_nu for n in range(len(aps))])
    return np.array(ret)

def _samestep(m1,m2,
              ev_tracks,flux_tracks,last_times,
              inits,distance,wav,aps,
              nsteps=1000):
    ev1,ev2 = ev_tracks[m1].copy(),ev_tracks[m2].copy()
    ev1,ev2 = ev1[:nsteps],ev2[:nsteps] #not sure why this is necessary, but...
    t1,t2 = last_times[m1],last_times[m2]
    dt1 = ev1['Time'][1] - ev1['Time'][0]
    dt2 = ev2['Time'][1] - ev2['Time'][0]
    
    fx1,fx2 = flux_tracks[m1].track.copy(),flux_tracks[m2].track.copy()
    
    fx1_steps = [row['Time'] in fx1['Time'] for row in ev1]
    fx2_steps = [row['Time'] in fx2['Time'] for row in ev2]
    either_steps = np.logical_or(fx1_steps,fx2_steps)
    first_step = np.argmax(either_steps)
    last_step = np.argmax(either_steps[::-1])

    #if table 1 starts first, add rows to table 2
    if np.argmax(fx2_steps) > first_step:
        nsteps = np.argmax(fx2_steps) - first_step
        fx2.reverse()
        add_t = fx2['Time'][-1]

        row_list = dict()
        for n in range(nsteps):
            row_to_add = [fx2['Time'][-1] - dt2,inits]
            row_list.update({row_to_add[0]:row_to_add[1]})
        add_table = Table([[key for key in row_list.keys()],
                           [val for val in row_list.values()]],
                          names=[*fx2.keys()])
        fx2 = vstack([fx2,add_table])
        fx2.reverse()
        del row_list,add_table

    #if table 2 starts first, add rows to table 1
    elif np.argmax(fx1_steps) > first_step:
        nsteps = np.argmax(fx1_steps) - first_step
        fx1.reverse()
        add_t = fx1['Time'][-1]

        row_list = dict()
        for n in range(nsteps):
            add_t -= dt1
            row_to_add = [add_t,inits]
            row_list.update({row_to_add[0]:row_to_add[1]})
        add_table = Table([[key for key in row_list.keys()],
                           [val for val in row_list.values()]],
                          names=[*fx1.keys()])
        fx1 = vstack([fx1,add_table])
        fx1.reverse()
        del row_list,add_table

    #if table 1 ends last, add rows to table 2
    if np.argmax(fx2_steps[::-1]) > last_step:
        nsteps = np.argmax(fx2_steps[::-1]) - last_step
        last_temp = ev2['Stellar_Temperature'][find_row(t2,ev2)] * u.K
        last_rad = ev2['Stellar_Radius'][find_row(t2,ev2)] * u.R_sun
        sed = make_bb(last_temp,last_rad,
                      distance,wav,aps)
        add_t = fx2['Time'][1]
        
        row_list = dict()
        for n in range(nsteps):
            add_t += dt2
            row_to_add = [add_t,sed]
            row_list.update({row_to_add[0]:row_to_add[1]})
        add_table = Table([[key for key in row_list.keys()],
                           [val for val in row_list.values()]],
                          names=[*fx2.keys()])
        fx2 = vstack([fx2,add_table])
        del row_list,add_table
        
    #elif table 2 ends last, add rows to table 1
    elif np.argmax(fx1_steps[::-1]) > last_step:
        nsteps = np.argmax(fx1_steps[::-1]) - last_step
        last_temp = ev1['Stellar_Temperature'][find_row(t1,ev1)] * u.K
        last_rad = ev1['Stellar_Radius'][find_row(t1,ev1)] * u.R_sun
        sed = make_bb(last_temp,last_rad,
                      distance,wav,aps)
        add_t = fx1['Time'][-1]
        
        row_list = dict()
        for n in range(nsteps):
            add_t += dt1
            row_to_add = [add_t,sed]
            row_list.update({row_to_add[0]:row_to_add[1]})
        add_table = Table([[key for key in row_list.keys()],
                           [val for val in row_list.values()]],
                          names=[*fx1.keys()])
        fx1 = vstack([fx1,add_table])
        del row_list,add_table
        
    return ev1,ev2,fx1,fx2
    
def _sametime(m1,m2,
              ev_tracks,flux_tracks,last_times,
              inits,distance,wav,aps):
    ev1,ev2 = ev_tracks[m1].copy(),ev_tracks[m2].copy()
    fx1,fx2 = flux_tracks[m1].track.copy(),flux_tracks[m2].track.copy()
    t1,t2 = last_times[m1],last_times[m2]
    
    overlap_12 = np.array([val in fx2['Time'] for val in fx1['Time']])
    overlap_21 = np.array([val in fx1['Time'] for val in fx2['Time']])

    #if table 1 starts first, add rows to table 2
    if np.argmin(overlap_12) < np.argmax(overlap_12):
        new_times = fx1['Time'][:np.argmax(overlap_12)][::-1]
        fx2.reverse()

        row_list = dict()
        for time in range(len(new_times)):
            row_to_add = [new_times[time],inits]
            row_list.update({row_to_add[0]:row_to_add[1]})
        add_table = Table([[key for key in row_list.keys()],
                           [val for val in row_list.values()]],
                          names=[*fx2.keys()])
        fx2 = vstack([fx2,add_table])
        fx2.reverse()
        del row_list,add_table

        #update overlap
        overlap_12 = np.array([val in fx2['Time'] for val in fx1['Time']])

    #else if table 2 starts first, add rows to table 1
    elif np.argmin(overlap_21) < np.argmax(overlap_21):
        new_times = fx2['Time'][:np.argmax(overlap_21)][::-1]
        fx1.reverse()

        row_list = dict()
        for time in range(len(new_times)):
            row_to_add = [new_times[time],inits]
            row_list.update({row_to_add[0]:row_to_add[1]})
        add_table = Table([[key for key in row_list.keys()],
                           [val for val in row_list.values()]],
                          names=[*fx1.keys()])
        fx1 = vstack([fx1,add_table])
        fx1.reverse()
        del row_list,add_table

        #update overlap
        overlap_21 = np.array([val in fx1['Time'] for val in fx2['Time']])

    #if table 1 ends last, add rows to table 2
    if np.argmin(overlap_12) > np.argmax(overlap_12):
        new_times = fx1['Time'][~overlap_12]
        last_temp = ev2['Stellar_Temperature'][find_row(t2,ev2)] * u.K
        last_rad = ev2['Stellar_Radius'][find_row(t2,ev2)] * u.R_sun
        sed = make_bb(last_temp,last_rad,
                      distance,wav,aps)

        row_list = dict()
        for time in new_times:
            row_to_add = [time,sed]
            row_list.update({row_to_add[0]:row_to_add[1]})
        add_table = Table([[key for key in row_list.keys()],
                           [val for val in row_list.values()]],
                          names=[*fx2.keys()])
        fx2 = vstack([fx2,add_table])
        del row_list,add_table

    #else if table 2 ends last, add rows to table 1
    elif np.argmin(overlap_21) > np.argmax(overlap_21):
        new_times = fx2['Time'][~overlap_21]
        last_temp = ev1['Stellar_Temperature'][find_row(t1,ev1)] * u.K
        last_rad = ev1['Stellar_Radius'][find_row(t1,ev1)] * u.R_sun
        sed = make_bb(last_temp,last_rad,
                      distance,wav,aps)

        row_list = dict()
        for time in new_times:
            row_to_add = [time,sed]
            row_list.update({row_to_add[0]:row_to_add[1]})
        add_table = Table([[key for key in row_list.keys()],
                           [val for val in row_list.values()]],
                          names=[*fx1.keys()])
        fx1 = vstack([fx1,add_table])
        del row_list,add_table

    return ev1,ev2,fx1,fx2

def standardize(m1,m2,
                ev_tracks,flux_tracks,
                last_times,history,
                inits,distance,wav,aps):
    samestep_hists = ['is','taper_is','taper_tc']
    sametime_hists = ['tc','ca','taper_ca','exp']

    if history in samestep_hists:
        ev1, ev2, fx1, fx2 = _samestep(m1,m2,
                                       ev_tracks,flux_tracks,last_times,
                                       inits,distance,wav,aps)
    elif history in sametime_hists:
        ev1, ev2, fx1, fx2 = _sametime(m1,m2,
                                       ev_tracks,flux_tracks,last_times,
                                       inits,distance,wav,aps)

    return ev1,ev2,fx1,fx2

def interp_tracks(mf,
                  masses,ev_tracks,flux_tracks,
                  last_temps,last_times,history,
                  inits,distance,wav,aps,
                  return_ev=False,return_flux=False):    
    # Retrieve the relevant tables (with modifications for interpolation)
    i = np.searchsorted(masses, mf)
    m1 = masses[i-1]
    m2 = masses[i]

    ev1, ev2, fx1, fx2 = standardize(m1,m2,
                                     ev_tracks,flux_tracks,
                                     last_times,history,
                                     inits,distance,wav,aps)
    
    frac = (mf - m1) / (m2 - m1)
    interp_fx = Table()
    for key in fx1.keys():
        interp_fx.add_column((1. - frac) * fx1[key] + frac * fx2[key],name=key)
    tf = (1. - frac) * last_times[m1] + frac * last_times[m2]
    rf = ((1. - frac) * ev1['Stellar_Radius'][find_row(tf,ev1)] + frac * ev2['Stellar_Radius'][find_row(tf,ev2)]) * u.R_sun
    tempf = (1. - frac) * last_temps[m1] + frac * last_temps[m2]
    interp_fx = interp_fx[interp_fx['Time'] < tf]
    
    row_to_add = [2 * interp_fx['Time'][0] - interp_fx['Time'][1],inits]
    interp_fx.reverse()
    interp_fx.add_row(row_to_add)
    interp_fx.reverse()

    sed = make_bb(tempf * u.K,rf,
                  distance,wav,aps)
    row_to_add = [tf,sed]
    interp_fx.add_row(row_to_add)
    
    interp_ev = Table()
    for key in ev1.keys():
        interp_ev.add_column((1. - frac) * ev1[key] + frac * ev2[key],name=key)
    first_time = np.argmin(abs(interp_ev['Time'] - interp_fx['Time'][0]))
    last_time = np.argmin(abs(interp_ev['Time'] - interp_fx['Time'][-1])) + 1
    interp_ev = interp_ev[first_time:last_time]

    if np.logical_and(return_ev,return_flux):
        return interp_ev,interp_fx
    elif return_ev:
        return interp_ev
    elif return_flux:
        return interp_fx
