
import pickle
import numpy as np
import matplotlib.pyplot as plt

import superposition_codes.auxiliary as spa
import general.plotting as gpl

def plot_code_sweeps(jobids, folder='superposition_codes/rf_sweeps/', axs=None,
                     fwid=3, data=None, err_ind=1, div_power=False, **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, 2, figsize=(2*fwid, fwid))
    if data is None:
        data = {}
    for ji in jobids:
        if data.get(ji) is None:
            s_ji = spa.load_sweeps(folder, ji)
            data[ji] = s_ji
        s_ji = data[ji]

        t = s_ji['args']['code_type']
        n_mods = s_ji['args']['n_modules']

        pwrs = s_ji['params'][0]
        if div_power:
            pwrs = pwrs/n_mods
        errs_pwr = s_ji['pwr_sweep'][err_ind]
        errs_pwr_theor = s_ji['pwr_sweep'][2]
        l = gpl.plot_trace_werr(np.sqrt(pwrs), errs_pwr[:, 0, 0].T, ax=axs[0],
                                conf95=True, 
                                label='{}, M = {}'.format(t, n_mods), **kwargs)
        color = l[0].get_color()
        axs[0].plot(np.sqrt(pwrs), errs_pwr_theor[:, 0, 0], color=color)

        nus = s_ji['params'][1]
        errs_nu = s_ji['nus_sweep'][err_ind]
        errs_nu_theor = s_ji['nus_sweep'][2]
        l = gpl.plot_trace_werr(nus, errs_nu[0, :, 0].T, ax=axs[1],
                                conf95=True, 
                                label='{}, M = {}'.format(t, n_mods), **kwargs)
        color = l[0].get_color()
        axs[1].plot(nus, errs_nu_theor[0, :, 0], color=color)

    return data, axs
