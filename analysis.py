
import numpy as np

import general.utility as u 

def orientation_centroids(stims, resp, exclude_end=True):
    u_oris = np.unique(stims)[:-1]
    rep = np.zeros((len(u_oris), resp.shape[1]))
    for i, uo in enumerate(u_oris):
        rep[i] = np.nanmean(resp[uo == stims], axis=0)
    return u_oris, rep

def noise_projection(stims, resp, exclude_half=False, **kwargs):
    rng = np.random.default_rng()
    u_oris, rep = orientation_centroids(stims, resp, **kwargs)
    if exclude_half:
        mask = u_oris < 180
        u_oris = u_oris[mask]
        rep = rep[mask]
    dreps = np.diff(rep, axis=0)
    proj_ons = []
    proj_offs = []
    proj_rands = []
    for i, uo in enumerate(u_oris[1:-1]):
        rand_vec = u.make_unit_vector(rng.normal(0, 1, (1, rep.shape[1])))
        sig_vec = u.make_unit_vector(np.mean(dreps[i:i+2], axis=0, keepdims=True))
        
        cent_vec = u.make_unit_vector(rep[i])
        mask = uo == stims

        cent = rep[i+1:i+2]
        proj_on = np.sum(sig_vec*(resp[mask] - cent), axis=1)
        proj_off = np.sum(cent_vec*(resp[mask] - cent), axis=1)
        proj_rand = np.sum(rand_vec*(resp[mask] - cent), axis=1)
        proj_ons.extend(proj_on)
        proj_offs.extend(proj_off)
        proj_rands.extend(proj_rand)
    return proj_ons, proj_offs, proj_rands
