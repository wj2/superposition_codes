
import functools as ft
import copy
import numpy as np
import scipy.linalg as sla
import scipy.special as ss
import scipy.stats as sts
import scipy.integrate as spi
import scipy.optimize as sopt
import sklearn.svm as skm
import sklearn.linear_model as sklm

import general.utility as u


def gain_decoding(
        stim,
        reps,
        orig_vec=None,
        offset=0,
        orig_ind=1,
        use_orig_dim=True,
        targ_ind=0,
        dim_mask=None,
):
    if orig_vec is None:
        if use_orig_dim:
            m = sklm.Ridge()
            targ = stim[:, orig_ind]
            m.fit(reps, targ - np.mean(targ))
            orig_vec = m.coef_
            offset = m.intercept_
            if dim_mask is not None:
                orig_vec = orig_vec * dim_mask
        else:
            orig_vec = sts.norm(0, 1).rvs((1, reps.shape[1]))
            orig_vec = u.make_unit_vector(orig_vec)
    
    targ = stim[:, targ_ind]
    targ = targ - np.mean(targ)
    mat = reps * orig_vec
    gains, err = sopt.nnls(mat, targ)
    
    gain_mat = np.identity(reps.shape[1]) * gains
    return gain_mat, orig_vec, err / stim.shape[0]
    

def orientation_centroids(stims, resp, exclude_end=True):
    u_oris = np.unique(stims)[:-1]
    rep = np.zeros((len(u_oris), resp.shape[1]))
    for i, uo in enumerate(u_oris):
        rep[i] = np.nanmean(resp[uo == stims], axis=0)
    return u_oris, rep    


def _kf(x, y, p, w, n_neur, a_range=(0, 1)):

    w2 = w**2
    pterm = .5*np.sqrt(np.pi)*w*np.exp(-((x - y)**2) / (4*w2))
    erf_term = (
        ss.erf((x + y + 2*a_range[0])/(2*w))
        - ss.erf((x + y - 2*a_range[1])/(2*w))
    )
    return n_neur*(p**2)*pterm*erf_term


def _kf_interference(x, p, w, n_units, n_mods):
    w2 = w**2
    pre = w*n_units*(n_mods - 1)*np.sqrt(2*np.pi)*p**2
    erf_term = ss.erf((1 - x)/(np.sqrt(2)*w)) + ss.erf(x/(np.sqrt(2)*w))
    exp_term = (
        np.sqrt(np.pi/2)*w*ss.erf(1/(np.sqrt(2)*w))
        - (1 - np.exp(-1/(2*w2)))*w2)
    out = pre*erf_term*exp_term
    return out


def code_kernel_theory(code, n_stim=100, targ_stim=.5, code_ind=0, use_full=False):
    code_i = code.code_list[code_ind]
    p = code_i.rf_scale
    w = code_i.wid
    n_units = code_i.n_units
    xs = np.linspace(0, 1, n_stim)
    y = np.array([targ_stim])
    kernel_targ = _kf(xs, y, p, w, n_units)
    kernel_full = _kf(np.expand_dims(xs, 1), np.expand_dims(xs, 0), p, w, n_units)
    if use_full:
        n_mods = len(code.code_list)
        kt_add = _kf_interference(xs, p, w, n_units, n_mods)
        kernel_targ = kernel_targ + kt_add
    return xs, kernel_targ, kernel_full    


def sample_kernels(
        code,
        n_samps=30,
        targ_stim=.5,
        code_ind=0,
        n_stims=100,
        use_full=False,
        random_noise=True,
):
    code_targ = code.code_list[code_ind]
    stim, rep = code_targ.sample_reps(n_stims, add_noise=False)
    ref_stim = np.linspace(0, 1, n_stims)
    stim[:, 0] = ref_stim
    rep = code_targ.get_rep(stim, add_noise=False)
    
    if use_full:
        samp_stim, f_rep = code.sample_reps(n_samps)    
        samp_stim[:, code_ind] = targ_stim
        add_dims = f_rep.shape[1] - rep.shape[1]
        add_zs = np.zeros((rep.shape[0], add_dims))
        rep = np.concatenate((rep, add_zs), axis=1)
        temp = code.get_rep(samp_stim, add_noise=random_noise)
    else:
        samp_stim, f_rep = code_targ.sample_reps(n_samps)    
        samp_stim[:] = targ_stim
        temp = code_targ.get_rep(samp_stim, add_noise=random_noise)
    kernels = temp @ rep.T
    return ref_stim, kernels    


def code_kernel(code, n_stims=100, targ_stim=.5, code_ind=0):
    code = code.code_list[code_ind]
    stim, rep = code.sample_reps(n_stims, add_noise=False)
    ref_stim = np.linspace(0, 1, n_stims)
    stim[:, 0] = ref_stim
    rep = code.get_rep(stim, add_noise=False)
    temp = code.get_rep(np.array([targ_stim]), add_noise=False)
    kernel = np.sum(rep * temp, axis=1)
    return ref_stim, kernel


def sample_full_kernels(*args, **kwargs):
    return sample_kernels(*args, use_full=True, **kwargs)


def _make_discrim_vec(code, targ_code=0, targ_pt=.5, delta=.1):
    code_targ = code.code_list[targ_code]
    
    stim = code_targ.sample_stim(2)
    stim[0] = targ_pt - delta / 2
    stim[1] = targ_pt + delta / 2
    reps = code_targ.get_rep(stim, add_noise=False)
    vec = u.make_unit_vector(
        reps[0] - reps[1]
    )
    return vec    


def discrimination_attention(
        code, targ_code=0, change_codes=None, var=1, **kwargs,
):
    vec = _make_discrim_vec(code, **kwargs)
    vec = np.expand_dims(vec, 0)
    other_codes = copy.deepcopy(code.code_list)
    other_codes.pop(targ_code)
    if change_codes is None:
        manip_codes = other_codes
    else:
        manip_codes = list(other_codes[i] for i in change_codes)

    n_vars = len(manip_codes)
    d_mat = np.identity(n_vars) * np.sqrt(var)

    def min_func(*covs):
        l_ = np.tri(n_vars, k=-1)
        (i1, i2) = np.where(l_ == 1)
        l_[(i1, i2)] = covs
        l_[(i2, i1)] = covs
        l_ += d_mat 
        cov = l_
        # l1 = l_ @ d_mat
        # cov = l1 @ l1.T
        # print(l_)
        # print(l_.shape, d_mat.shape)
        # print(cov)
        # print(cov)

        distr = sts.multivariate_normal(np.ones(n_vars)*.5, cov)
        def integ_func(*vs):
            interference = 0
            for i, mc in enumerate(manip_codes):
                v_arr = np.array([vs[i]])
                r_i = mc.get_rep(v_arr, add_noise=False)
                interference += np.sum(vec * r_i)**2
            return distr.pdf(vs) * interference

        ranges = list(zip(np.zeros(n_vars), np.ones(n_vars)))
        out = spi.nquad(integ_func, ranges)
        return out[0]
    initial_guess = np.zeros(np.sum(np.tri(n_vars, k=-1) == 1)) + .2
    bounds = ((-var, var),)*len(initial_guess)
    out = sopt.minimize(min_func, initial_guess, bounds=bounds)
    orig_out = min_func(np.zeros(len(initial_guess)))
    return out, orig_out

def make_cov(n_mods, init_cov, *covs):
    l_ = np.tri(n_mods, k=-1)
    (i1, i2) = np.where(l_ == 1)
    l_[(i1, i2)] = covs
    l_ += init_cov
    cov = l_ @ l_.T
    return cov


def simplified_correlation_attention(code, n_mods=10, var=1, **kwargs):
    vec = _make_discrim_vec(code, **kwargs)
    mat = sts.norm(0, 1).rvs((vec.shape[0], n_mods))
    vm = np.expand_dims(vec, 0) @ mat
    init_cov = np.identity(n_mods) * var
    init_var = vm @ init_cov @ vm.T

    initial_guess = np.zeros(np.sum(np.tri(n_mods, k=-1) == 1)) 
    save_vars = [init_var]
    save_covs = [initial_guess]
    def min_func(*covs):
        cov = make_cov(n_mods, init_cov, *covs)

        var = vm @ cov @ vm.T
        return var

    def save_func(covs):
        cov = make_cov(n_mods, init_cov, *covs)

        var = vm @ cov @ vm.T
        save_vars.append(var)
        save_covs.append(covs)

    bounds = ((-var, var),)*len(initial_guess)
    opt_out = sopt.minimize(
        min_func, initial_guess, bounds=bounds, callback=save_func,
    )
    end_cov = make_cov(n_mods, init_cov, *opt_out.x)
    end_var = opt_out.fun

    save_covs = np.stack(
        list(make_cov(n_mods, init_cov, *x) for x in save_covs), axis=0,
    )
    out = {
        "init": (init_cov, init_var),
        "end": (end_cov, end_var),
        "sequence": (save_covs, np.array(save_vars))
    }
    
    return out


def range_task(bounds, stim, ax=0):
    rs = stim[:, ax]
    targ = np.logical_and(
        rs < bounds[1], rs > bounds[0]
    )
    return targ


def make_range_task(bounds, **kwargs):
    func = ft.partial(range_task, bounds, **kwargs)
    return func


def task_func_learning(
        code,
        task_func,
        n_samps=1000,
        with_noise=False,
        model=skm.LinearSVC,
        n_grid=100,
        **kwargs,
):
    stim, reps = code.sample_reps(n_samps, add_noise=with_noise)
    targ = task_func(stim)
    m = model(**kwargs)
    m.fit(reps, targ)

    grid_pts = np.linspace(0, 1, n_grid)
    stim = code.sample_stim(n_grid)
    stim[:, 0] = grid_pts
    reps_te = code.get_rep(stim, add_noise=with_noise)
    pred_val = m.predict(reps_te)
    targ_val = task_func(stim)
    return grid_pts, pred_val, targ_val


def simplified_correlations(*args, n_reps=10, **kwargs):
    comb_out = {
        "init": ([], []),
        "end": ([], []),
    }
    for i in range(n_reps):
        out = simplified_correlation_attention(*args, **kwargs)
        for k, (cov, var) in comb_out.items():
            cov_i, var_i = out[k]
            cov.append(cov_i)
            var.append(var_i)
    return comb_out


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
        sig_vec = u.make_unit_vector(np.mean(dreps[i : i + 2], axis=0, keepdims=True))

        cent_vec = u.make_unit_vector(rep[i])
        mask = uo == stims

        cent = rep[i + 1 : i + 2]
        proj_on = np.sum(sig_vec * (resp[mask] - cent), axis=1)
        proj_off = np.sum(cent_vec * (resp[mask] - cent), axis=1)
        proj_rand = np.sum(rand_vec * (resp[mask] - cent), axis=1)
        proj_ons.extend(proj_on)
        proj_offs.extend(proj_off)
        proj_rands.extend(proj_rand)
    return proj_ons, proj_offs, proj_rands


def get_code_intersections(code, n_samps=1000, concatenate=False, add_noise=False):
    c1, c2 = code.code_list[:2]
    s1, r1 = c1.sample_reps(n_samps, add_noise=add_noise)
    s2, r2 = c2.sample_reps(n_samps, add_noise=add_noise)
    if concatenate:
        z_add = np.zeros_like(r1)
        r1 = np.concatenate((r1, z_add), axis=1)
        r2 = np.concatenate((z_add, r2), axis=1)
    dots = np.sum(r1 * r2, axis=1)
    return dots


def code_rate_change(
    code,
    rate1,
    rate2,
    new_stim=None,
    add_noise=True,
    n_stim=5000,
    random_projection=False,
):
    if new_stim is None:
        new_stim = .5
    stim1 = code.sample_stim(n_stim, rate=rate1)
    stim2 = code.sample_stim(n_stim, rate=rate2)
    stim2[:, rate1:rate2] = new_stim

    rep1 = code.get_rep(stim1, add_noise=add_noise)
    rep2 = code.get_rep(stim2, add_noise=add_noise)
    if random_projection:
        rng = np.random.default_rng()
        mat = rng.normal(0, 1, size=(rep1.shape[1], rep1.shape[1]))
        mat = sla.orth(mat)
        rep1 = rep1 @ mat
        rep2 = rep2 @ mat
    return rep1, rep2
