import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

import sklearn.decomposition as skd
import sklearn.metrics.pairwise as skmp 

import superposition_codes.auxiliary as spa
import general.plotting as gpl



def visualize_rf(
    code,
    ind=0,
    extent=(0, 1),
    n_pts=20,
    add_bar=True,
    ax=None,
    n_ticks=3,
    cmap="Greys",
    lv1_color="k",
    lv2_color="k",
    bar_mag=.3,
    **kwargs
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    vs = np.linspace(*extent, n_pts)
    xs, ys = np.meshgrid(vs, vs)
    xs = np.reshape(xs, (-1, 1))
    ys = np.reshape(ys, (-1, 1))
    reps = code.get_rep(np.concatenate((xs, ys), axis=1), add_noise=False)
    ri = reps[:, ind]
    rf_map = np.reshape(ri, (n_pts, n_pts))
    gpl.pcolormesh(vs, vs, rf_map, ax=ax, cmap=cmap, rasterized=True, **kwargs)

    if add_bar:
        gpl.make_xaxis_scale_bar(
            ax, magnitude=bar_mag, label="LV1", double=False, color=lv1_color
        )
        gpl.make_yaxis_scale_bar(
            ax, magnitude=bar_mag, label="LV2", double=False, color=lv2_color
        )
    gpl.clean_plot(ax, 1)
    ax.set_aspect("equal")


def plot_code_distances(
    code,
    n_pts=50,
    ax=None,
    default_val=.5,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    pts = np.linspace(0, 1, n_pts)
    stim = code.sample_stim(n_pts**2)
    pts = np.array(list(it.product(pts, repeat=2)))

    stim[:, 0:2] = pts
    stim[:, 2:] = default_val

    reps = code.get_rep(stim, add_noise=False)
    rep_dist = skmp.euclidean_distances(reps)
    sd1 = skmp.euclidean_distances(stim[:, 0:1])
    sd2 = skmp.euclidean_distances(stim[:, 1:2])
    stim_dist = skmp.euclidean_distances(stim)

    d1 = sd1.flatten()
    d2 = sd2.flatten()
    dists = rep_dist.flatten()

    weights, xe, ye = np.histogram2d(d1, d2, weights=dists)
    norms, xe, ye = np.histogram2d(d1, d2, bins=(xe, ye),)

    avg_dists = weights/norms
    gpl.pcolormesh(xe[:-1], ye[:-1], avg_dists, ax=ax)
    return avg_dists

    
def plot_code_interference(
    code,
    extent=(.4, .6),
    n_pts=100,
    n_inters=10,
    ax=None,
    avg_val=.5,
    n_dim=2,
    feature_ind=0,
    sv_label="single variable",
    av_label="all variables",
    **kwargs,
):    
    avg_stim = np.linspace(*extent, n_pts)
    n_feats = len(code.code_list)
    stim_fixed = np.ones((n_pts, n_feats))
    stim_fixed[:, feature_ind] = avg_stim
    reps = code.get_rep(stim_fixed, add_noise=False)
    stim_interf = code.sample_stim(n_pts)
    stim_interf[:, feature_ind] = avg_stim
    reps_interf = code.get_rep(stim_interf, add_noise=False)

    ax, p = gpl.plot_highdim_trace(
        reps, dim_red_mean=False, ax=ax, n_dim=n_dim, label=sv_label, **kwargs
    )
    gpl.plot_highdim_trace(
        reps_interf,
        n_dim=n_dim,
        ax=ax,
        p=p,
        ls="dashed",
        label=av_label,
        plot_outline=True,
        **kwargs,
    )
    ax.legend(frameon=False)
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax)
    gpl.make_2d_bars(ax, bar_len=(1, 1))
    return ax


def plot_code_vis(
    code,
    extent=(.4, .6),
    main_mid=.5,
    additional_mids=(),
    n_pts=100,
    ax=None,
    colors=None,
    n_dim=3,
    mid_colors=None,
    bar_len=2,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if colors is None:
        colors = (None, None)
    if mid_colors is None:
        mid_colors = (colors[1],)*len(colors)

    p = np.linspace(*extent, n_pts)
    z = np.ones(n_pts)*main_mid
    p1 = np.stack((p, z), axis=1)
    p2 = np.stack((z, p), axis=1)

    r1_mc = code.get_rep(p1, add_noise=False)
    r2_mc = code.get_rep(p2, add_noise=False)
    _, trs = gpl.plot_highdim_trace(
        r1_mc, r2_mc, ax=ax, n_dim=n_dim, colors=colors,
    )
    for i, mid in enumerate(additional_mids):
        p = np.linspace(*extent, n_pts)
        z = np.ones(n_pts)*mid
        p2 = np.stack((z, p), axis=1)

        r2_mc = code.get_rep(p2, add_noise=False)
        _, trs = gpl.plot_highdim_trace(
            r2_mc, ax=ax, n_dim=n_dim, colors=(mid_colors[i],), p=trs,
        )

    if n_dim == 2:
        gpl.make_xaxis_scale_bar(ax, magnitude=1, label="PC 1")
        gpl.make_yaxis_scale_bar(ax, magnitude=1, label="PC 2", text_buff=.4)
        gpl.clean_plot(ax, 0)
    else:
        gpl.clean_3d_plot(ax)
        gpl.make_3d_bars(ax, bar_len=bar_len)


def plot_code_sweeps(
    jobids,
    folder="superposition_codes/rf_sweeps/",
    axs=None,
    fwid=3,
    data=None,
    err_ind=1,
    div_power=False,
    colors=None,
    **kwargs
):
    if colors is None:
        colors = {}
    if axs is None:
        f, axs = plt.subplots(1, 2, figsize=(2 * fwid, fwid))
    if data is None:
        data = {}
    for ji in jobids:
        if data.get(ji) is None:
            s_ji = spa.load_sweeps(folder, ji)
            data[ji] = s_ji
        s_ji = data[ji]

        t = s_ji["args"]["code_type"]
        n_mods = s_ji["args"]["n_modules"]

        pwrs = s_ji["params"][0]
        if div_power:
            pwrs = pwrs / n_mods
        errs_pwr = s_ji["pwr_sweep"][err_ind]
        errs_pwr_theor = s_ji["pwr_sweep"][2]
        l = gpl.plot_trace_werr(
            np.sqrt(pwrs),
            errs_pwr[:, 0, 0].T,
            ax=axs[0],
            conf95=True,
            label="{}".format(t),
            color=colors.get(t),
            **kwargs
        )
        color = l[0].get_color()
        gpl.plot_trace_werr(
            np.sqrt(pwrs),
            errs_pwr_theor[:, 0, 0],
            color=color,
            ax=axs[0],
            linestyle="dashed",
            plot_outline=True,
        )

        nus = s_ji["params"][1]
        errs_nu = s_ji["nus_sweep"][err_ind]
        errs_nu_theor = s_ji["nus_sweep"][2]
        l = gpl.plot_trace_werr(
            nus,
            errs_nu[0, :, 0].T,
            ax=axs[1],
            conf95=True,
            color=colors.get(t),
            label="{}".format(t),
            **kwargs
        )
        color = l[0].get_color()
        gpl.plot_trace_werr(
            nus,
            errs_nu_theor[0, :, 0],
            color=color,
            linestyle="dashed",
            plot_outline=True,
            ax=axs[1],
        )

    return data, axs


def fano_factor(x, **kwargs):
    return np.var(x, **kwargs)/np.mean(x, **kwargs)


def plot_metric_diff(
    reps1,
    reps2,
    cents,
    axs=None,
    wid=.2,
    targ=.5,
    untarg=.9,
    single_neuron=False,
    metric=np.std,
    **kwargs,
):
    if axs is None:
        f, axs = plt.subplots(1, 3, squeeze=True)
    ax_tuned, ax_untuned, ax_all = axs
    untuned = np.logical_and(cents > untarg - wid/2,
                             cents < untarg + wid/2)
    tuned = np.logical_and(cents > targ - wid/2,
                           cents < targ + wid/2)

    r1_all = metric(reps1, axis=0)
    r2_all = metric(reps2, axis=0)

    r1_t = r1_all[tuned]
    r1_ut = r1_all[untuned]

    r2_t = r2_all[tuned]
    r2_ut = r2_all[untuned]

    if single_neuron:
        r1_t = r1_t[:1]
        r2_t = r2_t[:1]
        r1_ut = r1_ut[:1]
        r2_ut = r2_ut[:1]

    gpl.plot_trace_werr(
        [0, 1], np.stack((r1_t, r2_t), axis=1), ax=ax_tuned, conf95=True, **kwargs,
    )
    gpl.plot_trace_werr(
        [0, 1], np.stack((r1_ut, r2_ut), axis=1), ax=ax_untuned, conf95=True, **kwargs,
    )
    r12_all = np.stack((r1_all, r2_all), axis=1)
    gpl.plot_trace_werr([0, 1], r12_all, ax=ax_all, conf95=True, **kwargs)
    return axs
