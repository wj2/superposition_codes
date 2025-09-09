import numpy as np
import scipy.linalg as sla
import scipy.stats as sts

import matplotlib.pyplot as plt
import sklearn.decomposition as skd

import general.plotting as gpl
import general.paper_utilities as pu
import general.utility as u
import general.rf_models as rfm
import superposition_codes.analysis as spa
import superposition_codes.codes as spc
import superposition_codes.visualization as spv
import superposition_codes.auxiliary as spx


config_path = "superposition_codes/figures.conf"

colors = (
    np.array(
        [
            (127, 205, 187),
            (65, 182, 196),
            (29, 145, 192),
            (34, 94, 168),
            (37, 52, 148),
            (8, 29, 88),
        ]
    )
    / 256
)


class SuperpositionFigure(pu.Figure):
    def _make_color_dict(self, ks):
        return self._make_param_dict(ks)

    def _make_param_dict(self, ks, add="_color", func=None):
        if func is None:
            func = self.params.getcolor
        color_dict = {}
        for k in ks:
            color_dict[k] = func(k + add)
        return color_dict

    def mc_cmap(self):
        cm = plt.get_cmap(self.params.get("modular_cm"))
        return cm

    @property
    def mc_color(self):
        cm = self.mc_cmap()
        pt = self.params.getfloat("color_pt")
        return cm(pt)

    def sc_cmap(self):
        cm = plt.get_cmap(self.params.get("superpos_cm"))
        return cm

    @property
    def sc_color(self):
        cm = self.sc_cmap()
        pt = self.params.getfloat("color_pt")
        return cm(pt)

    @property
    def exper_color(self):
        ec = self.params.getcolor("exper_color")
        return ec

    def make_superpos_code(self, rescaling=False, remake=False):
        if self.data.get("superpos_code") is None or remake:
            n_modules = self.params.getint("n_modules", 2)
            pwr = self.params.getfloat("power", 10)
            n_units = self.params.getint("n_units", 100)
            set_rate = self.params.getint("set_rate")

            if rescaling:
                sc = spc.RescalingSuperposCode(
                    n_modules, pwr, n_units, set_rate=set_rate
                )
            else:
                sc = spc.SuperposCode(n_modules, pwr, n_units)
            self.data["superpos_code"] = sc
        return self.data["superpos_code"]

    def make_modular_code(self, rescaling=False, remake=False):
        if self.data.get("modular_code") is None or remake:
            n_modules = self.params.getint("n_modules", 2)
            pwr = self.params.getfloat("power", 10)
            n_units = self.params.getint("n_units", 100)

            if rescaling:
                mc = spc.RescalingModularCode(n_modules, pwr, n_units)
            else:
                mc = spc.ModularCode(n_modules, pwr, n_units)
            self.data["modular_code"] = mc
        return self.data["modular_code"]


class LinearVsRFCodes(SuperpositionFigure):
    def __init__(self, fig_key="linear_vs_rf", colors=colors, **kwargs):
        fsize = (8, 4)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        code_eg_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 40, 0, 38, 10, 5)
        code_eg = self.get_axs(
            code_eg_grid, sharex="all", sharey="all", squeeze=True,
        )
        
        code_pop_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 45, 100, 0, 45, 10, 0)
        pop_eg = self.get_axs(
            code_pop_grid, sharex="all", sharey="all", squeeze=True, all_3d=True
        )
        gss["panel_egs"] = (code_eg, pop_eg)

        recon_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 40, 45, 100, 10, 17)
        gss["panel_recon"] = self.get_axs(
            recon_grid, squeeze=True,
        )
        
        compute_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 60, 100, 50, 100, 10, 5)
        gss["panel_compute"] = self.get_axs(
            compute_grid, squeeze=True,
        )

        self.gss = gss

    def make_linear_code(self):
        if self.data.get("linear_code") is None:
            pwr = self.params.getfloat("power")
            n_units = self.params.getint("n_units")
            self.data["linear_code"] = spc.LinearCode(pwr, n_units)
        return self.data["linear_code"]

    def make_rf_code(self):
        if self.data.get("rf_code") is None:
            pwr = self.params.getfloat("power")
            n_units = self.params.getint("n_units")
            self.data["rf_code"] = spc.Code(pwr, n_units)
        return self.data["rf_code"]

    def panel_egs(self):
        key = "panel_egs"
        (ax_lin, ax_rf), (ax_pop_lin, ax_pop_rf) = self.gss[key]
        l_code = self.make_linear_code()
        rf_code = self.make_rf_code()

        n_pts = self.params.getint("n_pts")
        n_egs = self.params.getint("n_egs")
        l_cm = self.params.get("l_cm")
        l_cm = plt.get_cmap(l_cm)
        rf_cm = self.params.get("rf_cm")
        rf_cm = plt.get_cmap(rf_cm)
        l_colors = l_cm(np.linspace(.3, .8, n_egs))
        rf_colors = rf_cm(np.linspace(.3, .8, n_egs))

        xs = np.expand_dims(np.linspace(0, 1, n_pts), 1)
        l_rep = l_code.get_rep(xs, add_noise=False)
        rf_rep = rf_code.get_rep(xs, add_noise=False)

        rand_slopes = sts.norm(.3, .2).rvs(n_egs)
        rand_slopes[rand_slopes < 0] = 0

        for i in range(n_egs):
            ax_rf.plot(xs, rf_rep[:, i], color=rf_colors[i])
            ax_lin.plot(xs, l_rep[:, i] * rand_slopes[i], color=l_colors[i])
        gpl.clean_plot(ax_rf, 1)
        gpl.clean_plot(ax_lin, 0)
        ax_rf.set_xlabel("stimulus")
        ax_lin.set_xlabel("stimulus")
        ax_lin.set_ylabel("response")

        gpl.plot_highdim_trace(
            l_rep[20:80], ax=ax_pop_lin, dim_red_mean=False, colors=(l_cm(.5),)
        )

        gpl.plot_highdim_trace(
            rf_rep[20:80, :100], ax=ax_pop_rf, dim_red_mean=False, colors=(rf_cm(.5),)
        )

        gpl.make_3d_bars(ax_pop_lin, bar_len=1)
        gpl.make_3d_bars(ax_pop_rf, bar_len=1)

        ax_pop_rf.view_init(30, 45)
        ax_pop_lin.view_init(30, 50)

    def panel_recon(self):
        key = "panel_recon"
        (ax_dim, ax_err) = self.gss[key]

        n_units_range = self.params.getlist("n_units_range", typefunc=int)
        n_units_pts = self.params.getint("n_units_pts")
        n_units = np.linspace(*n_units_range, n_units_pts, dtype=int)
        pwr_per_unit = self.params.getfloat("pwr_per_unit")
        if self.data.get(key) is None:
            len_rf = np.zeros(n_units_pts)
            len_l = np.zeros_like(len_rf)
            mse_l = np.zeros_like(len_rf)
            mse_rf = np.zeros_like(len_rf)
            rf_wid = np.zeros_like(len_rf)
            for i, nu in enumerate(n_units):
                pwr_i = nu * pwr_per_unit
                rf_code = spc.Code(pwr_i, nu)
                l_code = spc.LinearCode(pwr_i, nu)
                mse_l[i] = l_code.get_predicted_mse()
                mse_rf[i] = rf_code.get_predicted_mse()
                rf_wid[i] = rf_code.wid

                len_rf[i] = np.sqrt(
                    rf_code.get_predicted_fi()
                )
                len_l[i] = np.sqrt(
                    l_code.get_empirical_fi_prediction()
                )
                
            self.data[key] = (
                (len_rf, len_l),
                (rf_wid, None),
                (mse_rf, mse_l),
            )

        lens, wids, mse = self.data[key]
        l_color = self.params.getcmap("l_cm")(.5)
        rf_cm = self.params.getcmap("rf_cm")
        rf_color = rf_cm(.5)
        rf_wid_color = rf_cm(.9)
        
        ax_dim.plot(n_units, lens[0], color=rf_color, label="RF expansion")
        ax_dim.plot(n_units, lens[1], color=l_color, label="linear expansion")
        ax_dim.set_yscale("log")

        wid_ax = ax_dim.twinx()
        wid_ax.set_yscale("log")
        wid_ax.plot(n_units, wids[0], color=rf_wid_color, label="RF width")
        wid_ax.set_ylabel("RF width", color=rf_wid_color)
        wid_ax.tick_params(axis="y", color=rf_wid_color)
        ax_dim.set_ylabel("expansion length")
        ax_dim.set_xlabel("number of units")
        gpl.clean_plot(ax_dim, 0)
                
        ax_err.plot(n_units, mse[0], color=rf_color)
        ax_err.plot(n_units, mse[1], color=l_color)
        ax_err.set_yscale("log")
        ax_err.set_ylabel("reconstruction error (MSE)")
        ax_err.set_xlabel("number of units")
        gpl.clean_plot(ax_err, 0)

    def panel_compute(self):
        key = "panel_compute"
        ax1, ax2 = self.gss[key]

        if self.data.get(key) is None:
            l_code = self.make_linear_code()
            rf_code = self.make_rf_code()

            tf = spa.make_range_task((.4, .6))
            stim, pred_rf, targ = spa.task_func_learning(rf_code, tf)
            stim, pred_l, _ = spa.task_func_learning(l_code, tf)
            self.data[key] = stim, targ, (pred_rf, pred_l)
        stim, targ, (pred_rf, pred_l) = self.data[key]
        
        lin_color = self.params.getcmap("l_cm")(.5)
        rf_color = self.params.getcmap("rf_cm")(.5)

        ax1.plot(stim, pred_rf, color=rf_color, label="RF code")
        ax2.plot(stim, pred_l, color=lin_color, label="linear code")

        gpl.plot_trace_werr(
            stim, targ, plot_outline=True, color="r", ls="dashed", ax=ax1, label="target"
        )
        gpl.plot_trace_werr(
            stim, targ, plot_outline=True, color="r", ls="dashed", ax=ax2, label="target"
        )

        gpl.clean_plot(ax2, 1)
        ax1.set_ylabel("prediction")
        ax1.set_xlabel("stimulus value")
        ax2.set_xlabel("stimulus value")
        

class SuperposEG(SuperpositionFigure):
    def __init__(self, fig_key="superpos_eg", colors=colors, **kwargs):
        fsize = (4, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        n_egs = self.params.getint("n_eg_units")

        code_eg1_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 45, 0, 80, 2, 2)
        code_eg1 = self.get_axs(
            code_eg1_grid, sharex="all", sharey="all", 
        )
        code_eg2_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 55, 100, 0, 80, 2, 2)
        code_eg2 = self.get_axs(
            code_eg2_grid, sharex="all", sharey="all", 
        )
        code_egs = (code_eg1, code_eg2)

        n_rows = int(np.ceil(n_egs / 2))
        m_map_grid = pu.make_mxn_gridspec(self.gs, n_rows, 2, 0, 45, 80, 100, 2, 2)
        m_map = self.get_axs(
            m_map_grid, sharex="all", sharey="all", 
        )        
        s_map_grid = pu.make_mxn_gridspec(self.gs, n_rows, 2, 55, 100, 80, 100, 2, 2)
        s_map = self.get_axs(
            s_map_grid, sharex="all", sharey="all", 
        )        
        gss["panel_egs"] = code_egs

        map_egs = (m_map.flatten(), s_map.flatten())
        gss["panel_map_egs"] = map_egs
        
        self.gss = gss

    def panel_egs(self):
        key = "panel_egs"
        axs_egs = self.gss[key]
        m_eg_axs, s_eg_axs = axs_egs

        n_egs = self.params.getint("n_eg_units")
        n_pts = self.params.getint("n_pts")
        stim = np.expand_dims(np.linspace(0, 1, n_pts), 1)

        m_code = self.make_modular_code()
        s_code = self.make_superpos_code()
        offsets = np.expand_dims(np.linspace(0, .5, n_egs), 0)
        colors = (self.params.getcolor("f1_color"), self.params.getcolor("f2_color"))
        for i, j in u.make_array_ind_iterator(m_eg_axs.shape):
            ind_slice = slice(i*n_egs, (i + 1)*n_egs)
            mod_code = m_code.code_list[j]
            sup_code = s_code.code_list[j]
            if i == j:
                m_reps = mod_code.get_rep(stim, add_noise=False)
                m_reps = m_reps[:, ind_slice]
            else:
                m_reps = np.zeros((n_pts, n_egs))
            s_reps = sup_code.get_rep(stim, add_noise=False)
            s_reps = s_reps[:, ind_slice]
            
            s_inds = np.argsort(np.squeeze(s_code.code_list[i].rf_cents)[ind_slice])
            s_eg_axs[i, j].plot(stim[:, 0], s_reps[:, s_inds] + offsets, color=colors[j])
            
            m_inds = np.argsort(np.squeeze(m_code.code_list[j].rf_cents)[ind_slice])
            m_eg_axs[i, j].plot(stim[:, 0], m_reps[:, m_inds] + offsets, color=colors[j])

            gpl.clean_plot(m_eg_axs[i, j], j)
            gpl.clean_plot(s_eg_axs[i, j], j)
            if i == 0:
                gpl.clean_plot_bottom(m_eg_axs[i, j])
                gpl.clean_plot_bottom(s_eg_axs[i, j])
            if j == 0:
                s_eg_axs[i, j].set_ylabel("response")
                m_eg_axs[i, j].set_ylabel("response")
            if i == 1:
                s_eg_axs[i, j].set_xlabel("LV {}".format(j + 1))
                m_eg_axs[i, j].set_xlabel("LV {}".format(j + 1))

    def panel_map_egs(self):
        key = "panel_map_egs"
        m_map_axs, s_map_axs = self.gss[key]

        m_code = self.make_modular_code()
        s_code = self.make_superpos_code()

        n_egs = self.params.getint("n_eg_units")
        colors = (self.params.getcolor("f1_color"), self.params.getcolor("f2_color"))

        n_neurs = int(m_code.n_units / len(m_code.code_list))
        m_half = np.arange(int(n_egs / 2))
        m_inds = np.concatenate((m_half, m_half + n_neurs))
        s_inds = np.arange(n_egs)

        for ax_i, k in enumerate(m_inds):
            spv.visualize_rf(
                m_code,
                ind=k,
                ax=m_map_axs[ax_i],
                lv1_color=colors[0],
                lv2_color=colors[1],
            )
        for ax_i, k in enumerate(s_inds):
            spv.visualize_rf(
                s_code,
                ind=k,
                ax=s_map_axs[ax_i],
                lv1_color=colors[0],
                lv2_color=colors[1],
            )
                

class InterferenceFigure(SuperpositionFigure):
    def __init__(self, fig_key="interference", colors=colors, **kwargs):
        fsize = (3, 3.7)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        interf_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 75, 0, 100, 10, 2)
        gss["panel_interference"] = self.get_axs(
            interf_grid,
        )

        gss["panel_alignment"] = self.get_axs((self.gs[78:100, 0:20],))[0, 0]
        self.gss = gss


    def panel_interference(self):
        key = "panel_interference"
        m_axs, s_axs = self.gss[key]
        sc = self.make_superpos_code()
        mc = self.make_modular_code()

        colors = (self.params.getcolor("f1_color"), self.params.getcolor("f2_color"))

        for i in range(len(m_axs)):
            spv.plot_code_interference(
                sc,
                ax=s_axs[i],
                color=colors[i],
                sv_label="LV {} changes".format(i + 1),
                av_label="LV 1 and 2 change",
            )
            spv.plot_code_interference(mc, ax=m_axs[i], color=colors[i])

    def panel_alignment(self):
        key = "panel_alignment"
        ax = self.gss[key]

        sc = self.make_superpos_code()
        mc = self.make_modular_code()

        n_pts = self.params.getint("n_pts")

        s1 = sc.sample_stim(n_pts)
        xs = np.linspace(0, 1, n_pts)
        stim1 = np.ones_like(s1) * .5
        stim1[:, 0] = xs
        stim2 = np.ones_like(s1) * .5
        stim2[:, 1] = xs
        s_r1 = sc.get_rep(stim1, add_noise=False)
        s_r2 = sc.get_rep(stim2, add_noise=False)
        s_ai = u.alignment_index(s_r1, s_r2)
        
        m_r1 = mc.get_rep(stim1, add_noise=False)
        m_r2 = mc.get_rep(stim2, add_noise=False)
        m_ai = u.alignment_index(m_r1, m_r2)

        ax.plot([0], [m_ai], "o", color=self.mc_color, ms=5)
        ax.plot([1], [s_ai], "o", color=self.sc_color, ms=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["modular", "superposition"], rotation=45)
        gpl.clean_plot(ax, 0)
        ax.set_ylabel("alignment index")
        ax.set_xlim([-.2, 1.2])
        gpl.add_hlines(0, ax)
        


class IntroductionFigure(SuperpositionFigure):
    def __init__(self, fig_key="intro", colors=colors, **kwargs):
        fsize = (8, 4)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        code_grid = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 0, 25, 10, 2)
        gss["panel_code_schematic"] = self.get_axs(
            code_grid, sharex="vertical", sharey="all"
        )

        pop_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 15, 35, 45, 75, 10, 5)
        pop_axs = self.get_axs(
            pop_grid,
            squeeze=True,
        )
        hist_grid = self.gs[10:35, 90:100]
        hist_ax = self.get_axs((hist_grid,))[0, 0]

        vis_grid = self.gs[0:45, 20:50]
        vis_ax = self.get_axs(
            (vis_grid,),
            all_3d=True,
        )[0, 0]
        gss["panel_pop_space"] = vis_ax, pop_axs, hist_ax

        vol_quant_grid = pu.make_mxn_gridspec(self.gs, 2, 3, 50, 100, 35, 100, 10, 10)
        vol_quant_ax = self.get_axs(vol_quant_grid, squeeze=True, sharex="all")
        gss["panel_volume"] = vol_quant_ax

        self.gss = gss

    def panel_code_schematic(self):
        key = "panel_code_schematic"
        axs = self.gss[key]
        sc = self.make_superpos_code()
        mc = self.make_modular_code()

        eg_stim = self.params.getlist("eg_stim", typefunc=float)
        stim = np.expand_dims(np.array(eg_stim), 0)
        reduce_fraction = 20

        sc_reps = []
        sc_cents = []
        mc_reps = []
        mc_cents = []
        for i, code in enumerate(sc.code_list):
            reps_i = code.get_rep(stim[:, i : i + 1], add_noise=False)
            cents_i = code.rf_cents
            sc_reps.append(np.squeeze(reps_i)[::reduce_fraction])
            sc_cents.append(np.squeeze(cents_i)[::reduce_fraction])

            mc_code = mc.code_list[i]
            mc_reps_i = mc_code.get_rep(stim[:, i : i + 1], add_noise=False)
            mc_reps.append(np.squeeze(mc_reps_i)[::reduce_fraction])
            mc_cents.append(np.squeeze(mc_code.rf_cents / 2 + i / 2)[::reduce_fraction])

        axs[0, 1].plot(sc_cents[0], sc_reps[0], "o", color=self.sc_color)
        axs[1, 1].plot(sc_cents[0], sc_reps[1], "o", color=self.sc_color)

        axs[0, 0].plot(mc_cents[0], mc_reps[0], "o", color=self.mc_color)
        axs[1, 0].plot(mc_cents[1], mc_reps[1], "o", color=self.mc_color)

        f1_color = self.params.getcolor("f1_color")
        f2_color = self.params.getcolor("f2_color")

        axs[1, 0].set_ylabel("response magnitude")
        gpl.clean_plot_bottom(axs[0, 0])
        gpl.clean_plot_bottom(axs[0, 1])
        gpl.clean_plot_bottom(axs[1, 0])
        gpl.clean_plot_bottom(axs[1, 1])
        gpl.clean_plot_bottom(axs[2, 0])
        gpl.clean_plot_bottom(axs[2, 1])

        gpl.add_hlines(-0.1, axs[0, 1], use_lim=(0, 1), color=f1_color)
        gpl.add_hlines(-0.1, axs[0, 0], use_lim=(0, 0.5), color=f1_color)
        gpl.add_hlines(-0.1, axs[1, 1], use_lim=(0, 1), color=f2_color)
        gpl.add_hlines(-0.1, axs[1, 0], use_lim=(0.5, 1), color=f2_color)

        gpl.add_hlines(-0.1, axs[2, 1], use_lim=(0, 1), color=f1_color)
        gpl.add_hlines(-0.2, axs[2, 1], use_lim=(0, 1), color=f2_color)
        gpl.add_hlines(-0.1, axs[2, 0], use_lim=(0, 0.5), color=f1_color)
        gpl.add_hlines(-0.1, axs[2, 0], use_lim=(0.5, 1), color=f2_color)

        sc_full_rep = sc.get_rep(stim, add_noise=False)[0]
        use_mc_cents = np.concatenate(mc_cents)
        mc_full_rep = np.concatenate(mc_reps)
        axs[2, 1].plot(
            sc_cents[0], sc_full_rep[::reduce_fraction], "o", color=self.sc_color,
        )
        axs[2, 0].plot(
            use_mc_cents, mc_full_rep, "o", color=self.mc_color
        )

        list(gpl.clean_plot(axs[i, 0], 0) for i in range(len(axs)))
        list(gpl.clean_plot(axs[i, 1], 1) for i in range(len(axs)))

    def panel_pop_space_nv(self):
        key = "panel_pop_space"
        ax_vis, axs_schem, ax_hist = self.gss[key]

        sc = self.make_superpos_code()
        mc = self.make_modular_code()

        n_pts = self.params.getint("n_pts")

        f1_color = self.params.getcolor("f1_color")
        f2_color = self.params.getcolor("f2_color")
        mid_colors = (
            gpl.add_color_value(f2_color, -0.2),
            gpl.add_color_value(f2_color, 0.2),
        )
        spv.plot_code_vis(
            sc,
            ax=ax_vis,
            n_pts=n_pts,
            n_dim=3,
            colors=(f1_color, f2_color),
            additional_mids=(0.4, 0.6),
            mid_colors=mid_colors,
        )

        spv.plot_code_interference(sc, ax=axs_schem[1], color=self.sc_color)
        spv.plot_code_interference(mc, ax=axs_schem[0], color=self.mc_color)

        if self.data.get(key) is None:
            sc_dots = spa.get_code_intersections(sc)
            mc_dots = spa.get_code_intersections(mc, concatenate=True)
            self.data[key] = (sc_dots, mc_dots)
        sc_dots, mc_dots = self.data[key]

        max_bin = np.max(np.concatenate((sc_dots, mc_dots)))
        bins = np.linspace(0, max_bin, 15)
        ax_hist.hist(sc_dots, bins=bins, color=self.sc_color, density=True)
        ax_hist.hist(mc_dots, bins=bins, color=self.mc_color, density=True)
        gpl.clean_plot(ax_hist, 0)
        gpl.make_yaxis_scale_bar(
            ax_hist,
            double=False,
            magnitude=1,
            label="density",
            text_buff=0.5,
        )
        ax_hist.set_xlabel("code interference")

    def panel_pop_space(self):
        key = "panel_pop_space"
        ax_vis, axs = self.gss[key]
        axs_schem = axs[:2]
        ax_hist = axs[2]

        sc = self.make_superpos_code()
        mc = self.make_modular_code()

        n_pts = self.params.getint("n_pts")

        f1_color = self.params.getcolor("f1_color")
        f2_color = self.params.getcolor("f2_color")
        mid_colors = (
            gpl.add_color_value(f2_color, -0.2),
            gpl.add_color_value(f2_color, 0.2),
        )
        spv.plot_code_vis(
            sc,
            ax=ax_vis,
            n_pts=n_pts,
            n_dim=3,
            colors=(f1_color, f2_color),
            additional_mids=(0.4, 0.6),
            mid_colors=mid_colors,
        )
        spv.plot_code_vis(
            sc,
            ax=axs_schem[1],
            n_pts=n_pts,
            n_dim=2,
            colors=(f1_color, f2_color),
        )
        spv.plot_code_vis(
            mc,
            ax=axs_schem[0],
            n_pts=n_pts,
            n_dim=2,
            colors=(f1_color, f2_color),
        )

        if self.data.get(key) is None:
            sc_dots = spa.get_code_intersections(sc)
            mc_dots = spa.get_code_intersections(mc, concatenate=True)
            self.data[key] = (sc_dots, mc_dots)
        sc_dots, mc_dots = self.data[key]

        max_bin = np.max(np.concatenate((sc_dots, mc_dots)))
        bins = np.linspace(0, max_bin, 15)
        ax_hist.hist(sc_dots, bins=bins, color=self.sc_color, density=True)
        ax_hist.hist(mc_dots, bins=bins, color=self.mc_color, density=True)
        gpl.clean_plot(ax_hist, 0)
        gpl.make_yaxis_scale_bar(
            ax_hist,
            double=False,
            magnitude=1,
            label="density",
            text_buff=0.5,
        )
        ax_hist.set_xlabel("code interference")

    def panel_volume(self):
        key = "panel_volume"
        ax_quant = self.gss[key]
        n_pts = self.params.getint("n_pts")
        n_samps = self.params.getint("n_samps")

        if self.data.get(key) is None:
            max_mod = self.params.getint("max_mod")
            mod_list = np.arange(1, max_mod + 1, dtype=int)
            scale_power = self.params.getfloat("scale_power")
            n_units = self.params.getint("n_units")

            skip = 0
            p = np.linspace(skip, 1 - skip, n_pts)
            z = np.ones(n_pts) * 0.5
            pts = np.stack((p, z), axis=1)

            sc_sig = np.zeros(len(mod_list))
            mc_sig = np.zeros(len(mod_list))
            sc_nu = np.zeros(len(mod_list))
            mc_nu = np.zeros(len(mod_list))

            sc_wid = np.zeros(len(mod_list))
            mc_wid = np.zeros(len(mod_list))

            sc_fi = np.zeros(len(mod_list))
            mc_fi = np.zeros(len(mod_list))

            sc_dist = np.zeros((len(mod_list)))
            mc_dist = np.zeros((len(mod_list)))

            sc_mse = np.zeros((len(mod_list)))
            mc_mse = np.zeros((len(mod_list)))
            
            sc_mse_emp = np.zeros((len(mod_list), n_samps))
            mc_mse_emp = np.zeros((len(mod_list), n_samps))
            for i, n_mod in enumerate(mod_list):
                sc_i = spc.SuperposCode(n_mod, scale_power * n_mod, n_units)
                sc_sig[i] = sc_i.sigma_est
                sc_wid[i] = sc_i.wid
                sc_fi[i] = sc_i.get_predicted_fi()
                sc_nu[i] = n_units
                sc_mse[i] = sc_i.get_predicted_mse()
                sc_mse_emp[i] = sc_i.empirical_mse(n_samps, boot=True)

                mc_i = spc.ModularCode(n_mod, scale_power * n_mod, n_units)
                mc_fi[i] = mc_i.get_predicted_fi()
                mc_wid[i] = mc_i.wid
                mc_nu[i] = n_units / n_mod
                mc_sig[i] = 1
                mc_mse[i] = mc_i.get_predicted_mse()
                mc_mse_emp[i] = mc_i.empirical_mse(n_samps, boot=True)

                rep_sc = sc_i.get_rep(pts, add_noise=False)
                rep_mc = mc_i.get_rep(pts, add_noise=False)

                sc_dist[i] = np.sum(
                    np.sqrt(np.sum(np.diff(rep_sc, axis=0) ** 2, axis=1, keepdims=True))
                )
                mc_dist[i] = np.sum(
                    np.sqrt(np.sum(np.diff(rep_mc, axis=0) ** 2, axis=1, keepdims=True))
                )

            # mc_dist_theor = (p[-1] - p[0]) * np.sqrt(
            #     rfm.random_uniform_fi_vec(scale_power, n_units / n_mod, mc_wid, 1)
            # )
            # sc_dist_theor = (p[-1] - p[0]) * np.sqrt(
            #     rfm.random_uniform_fi_vec(scale_power, n_units, sc_wid, 1)
            # )
            sc_pwr = scale_power * np.ones(len(mod_list))
            mc_pwr = sc_pwr

            self.data[key] = (
                mod_list,
                {
                    "power": (0, (sc_pwr, mc_pwr)),
                    "units": (1, (sc_nu, mc_nu)),
                    # "expansion": (2, (sc_dist_theor, mc_dist_theor)),
                    "width": (2, (sc_wid, mc_wid)), 
                    "effective": (3, (sc_sig, mc_sig)),
                    "Fisher": (4, (sc_fi, mc_fi)),
                    "error": (5, (sc_mse_emp.T, mc_mse_emp.T)),
                    "theory": (5, (sc_mse, mc_mse)),
                }
            )

        mod_list, quants = self.data[key]

        axs = list(ax_quant.T.flatten())
        y_labels = (
            "power\nper variable",
            "units\nper variable",
            # "expansion length",
            "RF width",
            "effective noise",
            "Fisher\ninformation",
            "error (MSE)",
        )
        style_kwargs = {
            "theory": {"plot_outline": True, "ls": "dashed", "log_y": True},
            "Fisher": {"log_y": True},
        }

        for k, quant in quants.items():
            i, (q_sc, q_mc) = quant
            ax = axs[i]
            sk = style_kwargs.get(k, {})
            gpl.plot_trace_werr(
                mod_list, q_sc, color=self.sc_color, ax=ax, conf95=True, **sk
            )
            gpl.plot_trace_werr(
                mod_list, q_mc, color=self.mc_color, ax=ax, conf95=True, **sk
            )
            gpl.clean_plot(ax, 0)
            ax.set_ylabel(y_labels[i])
            if i % 2 == 1:
                ax.set_xlabel("variables")
        ax.set_xticks([1, 5, 10, 15, 20])


class StatFigure(SuperpositionFigure):
    def __init__(self, fig_key="stat", colors=colors, **kwargs):
        fsize = (6, 2.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        code_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 45, 40, 100, 10, 10)
        axs = self.get_axs(
            code_grid,
            squeeze=True,
        )
        gss["panel_gratings_move"] = axs

        ax_vis = self.get_axs(
            (self.gs[:55, 0:50],),
            all_3d=True,
        )[0, 0]
        gss["panel_vis"] = ax_vis

        self.gss = gss

    def _get_exper_data(self, shift_amt=0):
        if self.data.get("exper_data") is None:
            manifest, data = spx.load_orientation_data()

            proc_data = {}
            for key, exper_data in data.items():
                exper_data = data[key]
                if shift_amt > 0:
                    exper_data = exper_data[0][:-shift_amt], exper_data[1][shift_amt:]
                stim, rep = spa.orientation_centroids(*exper_data)
                proc_data[key] = (shift_amt, exper_data, stim, rep)
            self.data["exper_data"] = proc_data
        return self.data["exper_data"]

    def _compute_exper_data_projections(self, recompute=False):
        if self.data.get("exper_projections") is None or recompute:
            eds = self._get_exper_data()
            out_projs = {}
            for k, out in eds.items():
                _, data1_shift, _, _ = out
                projs = spa.noise_projection(*data1_shift)
                out_projs[k] = projs
            self.data["exper_projections"] = out_projs
        return self.data["exper_projections"]

    def panel_vis(self, recompute=True):
        key = "panel_vis"
        ax_vis = self.gss[key]
        shift_dim = 0
        if self.data.get(key) is None or recompute:
            exp_d = self._get_exper_data()
            _, data1_shift, _, rep = list(exp_d.values())[0]
            p = skd.PCA(3 + shift_dim)
            rep_low = p.fit_transform(rep)
            rep_pts = p.transform(data1_shift[1])
            self.data[key] = (rep_low, rep_pts)
        rep_low, rep_pts = self.data[key]

        col_inds = np.tile(np.linspace(0, 1, 16), 2)

        gpl.plot_colored_line(
            rep_low[:, 0 + shift_dim],
            rep_low[:, 1 + shift_dim],
            rep_low[:, 2 + shift_dim],
            col_inds=col_inds,
            ax=ax_vis,
            cmap="hsv",
        )
        gpl.clean_3d_plot(ax_vis)
        gpl.make_3d_bars(ax_vis, bar_len=2000)
        ax_vis.view_init(30, 40)

    def panel_gratings_move(self, recompute=False):
        key = "panel_gratings_move"
        az_schem, ax_quant = self.gss[key]

        if self.data.get(key) is None or recompute:
            sc = self.make_superpos_code()
            mc = self.make_modular_code()
            codes = {"superposition": sc, "modular": mc}
            n_reps = self.params.getint("n_reps")
            n_pts = self.params.getint("n_pts")
            random_projection = self.params.getboolean("random_projection")
            use_noise = self.params.getboolean("use_noise")

            vals = np.linspace(0.4, 0.6, n_pts)
            vals = np.repeat(vals, n_reps)

            out = {}
            for k, code in codes.items():
                stim = code.sample_stim(len(vals))
                stim[:, 0] = vals
                reps = code.get_rep(stim, add_noise=use_noise)
                if random_projection:
                    mat = self.rng.normal(0, 1, size=(reps.shape[1], reps.shape[1]))
                    mat = sla.orth(mat)
                    reps = reps @ mat
                p_on, p_off, p_rand = spa.noise_projection(stim[:, 0], reps)
                out[k] = (p_on, p_off, p_rand)

            out.update(self._compute_exper_data_projections(recompute))
            self.data[key] = out
        out = self.data[key]

        colors = {
            "superposition": self.sc_color,
            "modular": self.mc_color,
            "experiment": self.exper_color,
        }
        label_dict = {
            "superposition": "superposition",
            "modular": "modular",
        }
        label_count = {}
        for k, (p_on, p_off, p_rand) in out.items():
            v = np.array([np.std(p_on), np.std(p_off), np.std(p_rand)]) ** 2
            # v = u.make_unit_vector(v)
            v = v / np.sum(v)
            label = label_dict.get(k, "experiment")
            count = label_count.get(label, 0)
            count += 1
            label_count[label] = count
            if count > 1:
                label = ""
            ax_quant.plot(
                [0, 1, 2],
                v,
                ms=5,
                label=label,
                alpha=0.5,
                color=colors.get(k, self.exper_color),
            )
            ax_quant.plot(
                [0, 1, 2],
                v,
                "o",
                ms=5,
                color=colors.get(k, self.exper_color),
            )
            ax_quant.legend(frameon=False)
        ax_quant.set_xticks([0, 1, 2])
        ax_quant.set_xticklabels(["coding", "origin", "random"])
        ax_quant.set_xlabel("directions")
        ax_quant.set_ylabel("proportion of variance")
        gpl.clean_plot(ax_quant, 0)


class WidthInterferenceFigure(SuperpositionFigure):
    def __init__(self, fig_key="width-interference", colors=colors, **kwargs):
        fsize = (1, .75)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        change_grid = pu.make_mxn_gridspec(self.gs, 1, 1, 0, 100, 0, 100, 10, 20)
        gss["panel_width_change"] = self.get_axs(
            change_grid,
        )[0, 0]

        self.gss = gss

    def panel_width_change(self):
        key = "panel_width_change"
        ax = self.gss[key]

        nu = self.params.getint("n_units")
        pwr = self.params.getfloat("power")
        total_feats = self.params.getint("total_features")
        sub_feats = self.params.getlist("feature_subsets", typefunc=int)

        s_cm = self.params.getcmap("superpos_cm")
        s_colors = s_cm(np.linspace(.3, .7, len(sub_feats)))

        wids = np.linspace(.001, .5, 100)
        for i, sb in enumerate(sub_feats):
            sc_i = spc.RescalingSuperposCode(total_feats, pwr, nu, set_rate=sb)
            sig = sc_i.sigma_est
            mses = rfm.mse_w_range(pwr / sb, nu, 1, wid=wids, sigma_n=sig)
            ax.plot(
                wids,
                mses,
                color=s_colors[i],
                label="{} / {} variables".format(sb, total_feats),
            )
        mse_mc = rfm.mse_w_range(pwr / total_feats, nu / total_feats, 1, wid=wids)
        ax.plot(wids, mse_mc, color=self.mc_color, ls="dashed")
        ax.set_yscale("log")
        gpl.clean_plot(ax, 0)
        ax.set_xlabel("RF width")
        ax.set_ylabel("MSE")
        ax.legend(frameon=False)


class SometimesFigure(SuperpositionFigure):
    def __init__(self, fig_key="sometimes", colors=colors, **kwargs):
        fsize = (5, 4)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        mse_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 40, 0, 100, 10, 20)
        gss["panel_mse_comparison"] = self.get_axs(
            mse_grid,
            squeeze=True,
            sharex="all",
        )

        data_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 60, 100, 0, 100, 10, 10)
        gss["panel_data_comparison"] = self.get_axs(
            data_grid,
            squeeze=True,
            sharex="all",
            sharey="all",
        )

        self.gss = gss

    def panel_mse_comparison(self):
        key = "panel_mse_comparison"
        ax_noise, ax_mse = self.gss[key]

        rmc = self.make_modular_code(rescaling=True)

        if self.data.get(key) is None:
            nm = self.params.getint("n_modules")
            pwr = self.params.getfloat("power")
            n_units = self.params.getint("n_units")

            rates = np.arange(1, nm + 1)[::-1]
            fractions = rates / nm
            set_rates = np.arange(10, nm, 10)
            mc_sig = list(rmc._rate_sigma(rate=r) for r in rates)
            mc_mse = list(rmc.get_predicted_mse(rate=r) for r in rates)
            sc_mse_all = np.zeros((len(set_rates), len(rates)))
            sc_sig_all = np.zeros((len(set_rates), len(rates)))
            for i, sr_i in enumerate(set_rates):
                rsc_i = spc.RescalingSuperposCode(nm, pwr, n_units, set_rate=sr_i)
                sc_mse_all[i] = list(rsc_i.get_predicted_mse(rate=r) for r in rates)
                sc_sig_all[i] = list(rsc_i._rate_sigma(rate=r) for r in rates)
            self.data[key] = (fractions, (mc_sig, mc_mse), (sc_sig_all, sc_mse_all))

        fractions, (mc_sig, mc_mse), (sc_sig_all, sc_mse_all) = self.data[key]
        sc_cmap = self.sc_cmap()

        ax_noise.plot(fractions, mc_sig, color=self.mc_color)
        ax_mse.plot(fractions, mc_mse, color=self.mc_color)
        gap = 0.4
        sc_cm_pts = np.linspace(gap, 1, len(sc_sig_all))
        for i, sc_i in enumerate(sc_sig_all):
            col = sc_cmap(sc_cm_pts[i])
            ax_noise.plot(fractions, sc_i, color=col)
            ax_mse.plot(fractions, sc_mse_all[i], color=col)

        ax_mse.set_yscale("log")
        ax_mse.set_xscale("log")
        ax_mse.set_xlabel("actual rate fraction")
        ax_noise.set_xlabel("actual rate fraction")
        ax_noise.set_ylabel("effective noise")
        ax_mse.set_ylabel("error (MSE)")
        gpl.clean_plot(ax_mse, 0)
        gpl.clean_plot(ax_noise, 0)

    def panel_data_comparison(self):
        key = "panel_data_comparison"
        axs = self.gss[key]

        rate1 = self.params.getint("rate")
        rate2 = rate1 + 1
        if self.data.get(key) is None:
            n_modules = self.params.getint("var_n_modules")
            pwr = self.params.getfloat("power")
            n_units = self.params.getint("n_units")
            random_projection = self.params.getboolean("random_projection")

            rmc = spc.RescalingModularCode(n_modules, pwr, n_units)
            rsc = spc.RescalingSuperposCode(n_modules, pwr, n_units)

            mc_r1, mc_r2 = spa.code_rate_change(
                rmc,
                rate1,
                rate2,
                random_projection=random_projection,
            )
            sc_r1, sc_r2 = spa.code_rate_change(rsc, rate1, rate2)

            self.data[key] = ((rmc, mc_r1, mc_r2), (rsc, sc_r1, sc_r2))
        (rmc, mc_r1, mc_r2), (rsc, sc_r1, sc_r2) = self.data[key]
        wid = self.params.getfloat("tune_wid")

        spv.plot_metric_diff(
            mc_r1,
            mc_r2,
            rmc.rf_cents[:, rate1],
            axs=axs,
            wid=wid,
            color=self.mc_color,
        )
        spv.plot_metric_diff(
            sc_r1,
            sc_r2,
            rsc.rf_cents[:, rate1],
            axs=axs,
            wid=wid,
            color=self.sc_color,
        )

        axs[0].set_xticks([0, 1])
        axs[0].set_xticklabels(["pre-stimulus", "post-stimulus"])
        axs[0].set_xlim([-0.2, 1.2])
        axs[0].set_ylabel("response standard deviation")
        axs[0].set_title("preferred stimulus")
        axs[1].set_title("non-preferred stimulus")
        axs[2].set_title("all units")
