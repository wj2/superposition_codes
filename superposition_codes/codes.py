import numpy as np
import scipy.stats as sts
import sklearn.linear_model as sklm
import sklearn.svm as skm
import joblib as jl
import scipy.linalg as sla

import general.rf_models as rfm
import general.utility as u


def make_orthonormal_matrix(n_dim):
    rng = np.random.default_rng()
    mat = rng.normal(0, 1, size=(n_dim, n_dim))
    mat = sla.orth(mat)
    return mat


class Code:
    def __init__(
        self,
        pwr,
        n_units,
        dims=1,
        wid=None,
        lam=2,
        spk_cost="l2",
        pop_func="random",
        sigma_n=1,
        modules=1,
        make_code=True,
        use_ramp=None,
        center_at_zero=False,
    ):
        if spk_cost == "l1":
            def cost_func(x): return np.mean(np.sum(np.abs(x), axis=1))
            def titrate_func(x, y): return x / y
        elif spk_cost == "l2":
            def cost_func(x): return np.mean(np.sum(x**2, axis=1))
            def titrate_func(x, y): return np.sqrt(x / y)
        else:
            raise IOError("unrecognized spk_cost indicator {}".format(spk_cost))
        if pop_func == "random":
            pop_func = rfm.get_random_uniform_pop
        elif pop_func == "lattice":
            pop_func = rfm.get_lattice_uniform_pop
        else:
            raise IOError("unrecognized pop_func indicator {}".format(pop_func))

        if wid is None:
            out = rfm.min_mse_vec(
                pwr, n_units, dims, ret_components=True, sigma_n=sigma_n, lam=lam
            )
            wid = out[-1]

        pwr_pre = rfm.random_uniform_pwr(n_units, wid, dims, scale=1)
        rescale = np.sqrt(pwr / pwr_pre)
        self.rf_scale = rescale
        if make_code:
            out = pop_func(
                pwr,
                n_units,
                dims,
                w_use=wid,
                sigma_n=sigma_n,
                cost_func=cost_func,
                titrate_func=titrate_func,
                scale_use=rescale,
                ret_params=True,
                use_ramp=use_ramp,
            )
            stim_distr, rf, drf, noise, ms, ws = out

            self.rf_cents = ms
            self.stim_distr = stim_distr
            self.rf = rf
            self.drf = drf
            self.noise = noise

        self.use_ramp = use_ramp
        self.n_units = n_units
        self.pwr = pwr
        self.dims = dims
        self.wid = wid
        self.sigma_n = sigma_n

        self.linear_decoder = None
        if center_at_zero:
            mean_interference = rfm.random_uniform_unit_mean(
                wid, dims, rescale,
            )
            self.caz = -mean_interference
        else:
            self.caz = 0
        self.add_dc = 0

    # fix this
    def get_empirical_thr_error_pred(self, n_samps=1000, lam=2):
        mag = 1 / 6
        stim1 = self.stim_distr.rvs(n_samps)
        stim2 = self.stim_distr.rvs(n_samps)
        d12 = np.sqrt(np.sum((stim1 - stim2) ** 2, axis=1))
        mask = d12 > self.wid * lam
        stim1 = stim1[mask]
        stim2 = stim2[mask]
        s1 = self.rf(stim1)
        s2 = self.rf(stim2)
        dists = np.sum((s1 - s2) ** 2, axis=1)
        use_dist = np.mean(dists) - lam * np.std(dists)
        use_dist = np.sqrt(max(use_dist, 0))

        prob = np.mean(sts.norm(0, 1).cdf(-np.sqrt(dists) / (2 * self.sigma_n)))
        if self.use_ramp is not None:
            gd = self.dims - len(self.use_ramp)
        else:
            gd = self.dims

        factor = min(1 / (2 * self.wid) ** gd, self.n_units)
        p = factor * prob
        return p, mag

    def get_empirical_fi_prediction(self, n_samps=1000):
        stim = self.stim_distr.rvs(n_samps)
        fi = (self.drf(stim) ** 2) / (self.sigma_n**2)
        fi_mse = 1 / np.sum(np.mean(fi, axis=0), axis=0)
        thr_prob, mag = self.get_empirical_thr_error_pred(n_samps)
        thr_mse = thr_prob * mag
        out = fi_mse + thr_mse
        return out

    def get_predicted_fi(self, **kwargs):
        use_n_units = kwargs.get("n_units", self.n_units)
        use_pwr = kwargs.get("pwr", self.pwr)
        use_wid = kwargs.get("wid", self.wid)
        use_dims = kwargs.get("dims", self.dims)
        use_sigma_n = kwargs.get("sigma_n", self.sigma_n)
        return rfm.random_uniform_fi_pwr(
            use_n_units, use_pwr, use_wid, use_dims, sigma_n=use_sigma_n,
        )

    def get_predicted_threshold_err(self, **kwargs):
        use_n_units = kwargs.get("n_units", self.n_units)
        use_pwr = kwargs.get("pwr", self.pwr)
        use_wid = kwargs.get("wid", self.wid)
        use_dims = kwargs.get("dims", self.dims)
        use_sigma_n = kwargs.get("sigma_n", self.sigma_n)
        # pwr_var = self.get_activity_var()

        p, mag = rfm.compute_threshold_vec(
            use_pwr, use_n_units, use_dims, use_wid, sigma_n=use_sigma_n,
        )
        return p, mag

    def get_predicted_mse(self, **kwargs):
        fi = self.get_predicted_fi(**kwargs)
        t_prob, t_mag = self.get_predicted_threshold_err(**kwargs)
        return t_prob * t_mag + (1 - t_prob) * (1 / fi)

    def get_activity_var(self):
        pwr_var = rfm.random_uniform_pwr_var_fix(
            self.n_units, self.pwr, self.wid, self.dims
        )
        return pwr_var

    def get_predicted_fi_var(self):
        return rfm.random_uniform_fi_var_pwr(
            self.n_units, self.pwr, self.wid, self.dims
        )

    def generate_errors(self, n_samps=1000, eps=0.0001):
        p = self.get_predicted_threshold_err()[0]
        fi = self.get_predicted_fi()
        fi_var = self.get_predicted_fi_var()
        rng = np.random.default_rng()
        fis = rng.normal(fi, np.sqrt(fi_var), size=(n_samps,))
        fis[fis < 0] = np.nan

        thr_err = np.abs(
            rng.uniform(0, 1, size=n_samps) - rng.uniform(0, 1, size=n_samps)
        )
        trl_type = rng.uniform(0, 1, size=n_samps) < p
        out = np.zeros(n_samps)
        out[trl_type] = thr_err[trl_type]
        out[np.logical_not(trl_type)] = 1 / fis[np.logical_not(trl_type)]
        return out

    def sample_stim(self, n_stim=1000):
        return self.stim_distr.rvs(n_stim)

    def sample_reps(self, n_reps=1000, add_noise=True, **kwargs):
        stim = self.sample_stim(n_reps, **kwargs)
        reps = self.get_rep(stim, add_noise=add_noise)
        return stim, reps

    def get_rep(self, stim, add_noise=True):
        rep = self.rf(stim) + self.caz
        if add_noise:
            rep = rep + self.noise.rvs(rep.shape)
        return rep

    def decode_rep_brute(self, rep, n_candidates=500):
        return rfm.brute_decode_rf(
            rep, self.get_rep, self.dims, n_gran=n_candidates, add_noise=False
        )

    def decode_rep_refine(self, rep, n_candidates=10):
        return rfm.refine_decode_rf(
            rep, self.get_rep, self.dims, n_gran=n_candidates, add_noise=False
        )

    def decode_rep_brute_decoupled(self, rep, dim_i=(0,), n_candidates=2000):
        return rfm.brute_decode_decouple(
            rep,
            self.get_rep,
            self.dims,
            dim_i=dim_i,
            n_gran=n_candidates,
            add_noise=False,
            add_dc=self.add_dc,
        )

    def train_linear_decoder(
        self, model=sklm.Ridge, n_training=1000, dim_i=None, **kwargs
    ):
        stim, reps = self.sample_reps(n_training)
        if dim_i is not None:
            stim = stim[:, dim_i]
        stim = np.squeeze(stim)
        m = model(**kwargs)
        m.fit(reps, stim)
        return m

    def decode_rep_linear(self, rep, n_training=10000, **kwargs):
        if self.linear_decoder is None:
            self.linear_decoder = self.train_linear_decoder(**kwargs)
        pred = self.linear_decoder.predict(rep)
        if len(pred.shape) == 1:
            pred = np.expand_dims(pred, 1)
        return pred

    def decode_rep(self, rep, method="brute", **kwargs):
        if method == "brute":
            out = self.decode_rep_brute(rep, **kwargs)
        elif method == "brute_decoupled":
            out = self.decode_rep_brute_decoupled(rep, **kwargs)
        elif method == "linear":
            out = self.decode_rep_linear(rep, **kwargs)
        elif method == "kernel":
            out = self.decode_rep_linear(rep, model=skm.SVR, **kwargs)
        elif method == "refine":
            out = self.decode_rep_refine(rep, **kwargs)
        else:
            raise IOError("unrecognized decoding method")
        return out

    def empirical_mse(self, n_samps=2000, boot=False, single_dim=0, **kwargs):
        stim, rep = self.sample_reps(n_samps)
        stim_hat = self.decode_rep(rep, **kwargs)
        if single_dim is not None:
            stim = stim[:, single_dim]
            stim_hat = stim_hat[:, single_dim]

        mse = (stim - stim_hat) ** 2
        if boot:
            mse = u.bootstrap_list(mse, np.mean, n=n_samps)
        return mse


class LinearCode(Code):
    def __init__(self, *args, dims=1, **kwargs):
        use_ramp = np.arange(dims)
        super().__init__(*args, dims=dims, use_ramp=use_ramp, **kwargs)

    def get_empirical_fi_prediction(self, n_samps=1000):
        stim = self.stim_distr.rvs(n_samps)
        fi = (self.drf(stim) ** 2) / (self.sigma_n**2)
        fi = np.sum(np.mean(fi, axis=0), axis=0)
        return fi

    def get_predicted_mse(self, **kwargs):
        return 1 / self.get_empirical_fi_prediction(**kwargs)


class MultiCode(Code):
    def sample_stim(self, n_stim=1000):
        if self.stim_distr is None:
            out = self._sample_stim(n_stim=n_stim)
        else:
            out = self.stim_distr.rvs(n_stim)
        return out

    def _sample_stim(self, n_stim=1000):
        out = np.zeros((n_stim, self.dims))
        for i, code in enumerate(self.code_list):
            sd_b = i * self.dims_per_module
            sd_e = (i + 1) * self.dims_per_module
            out[:, sd_b:sd_e] = code.sample_stim(n_stim)
        return out

    def get_rep(self, stim, add_noise=True, combine=np.nansum):
        rep = np.zeros((stim.shape[0], self.n_units_per_module, self.n_modules))
        for i, code in enumerate(self.code_list):
            sd_b = i * self.dims_per_module
            sd_e = (i + 1) * self.dims_per_module
            rep[..., i] = code.get_rep(stim[:, sd_b:sd_e], add_noise=False)
        rep = combine(rep, axis=2)
        if add_noise:
            rep = rep + self.noise.rvs(rep.shape)
        if self.linear_transform is not None:
            rep = rep @ self.linear_transform
        return rep

    def decode_rep(self, rep, method="brute_decoupled", **kwargs):
        dim_i = np.arange(self.dims_per_module, dtype=int)
        return super().decode_rep(rep, method=method, dim_i=dim_i, **kwargs)

    def get_predicted_mse(self, code_ind=0):
        return self.code_list[code_ind].get_predicted_mse()

    def get_predicted_fi(self, code_ind=0):
        return self.code_list[code_ind].get_predicted_fi()

    @property
    def capacity(self):
        stim, reps = self.sample_reps(add_noise=False)
        sig_var = np.mean(np.var(reps, axis=0))
        ratio = sig_var / (self.sigma_n**2)
        capacity = self.n_units * 0.5 * np.log(1 + ratio)
        return capacity


class RescalingMultiCode(MultiCode):
    def sample_stim(self, n_stim=1000, rate=None):
        if rate is None:
            rate = self.n_modules
        out = np.zeros((n_stim, self.dims))
        for i, code in enumerate(self.code_list):
            sd_b = i * self.dims_per_module
            sd_e = (i + 1) * self.dims_per_module
            stim = code.sample_stim(n_stim)
            if i > rate - 1:
                stim = stim*np.nan
            out[:, sd_b:sd_e] = stim
        return out

    def get_rep(self, stim, add_noise=True, combine=np.nansum):
        rep = np.zeros((stim.shape[0], self.n_units_per_module, self.n_modules))
        off = np.zeros((stim.shape[0], 1))
        for i, code in enumerate(self.code_list):
            sd_b = i * self.dims_per_module
            sd_e = (i + 1) * self.dims_per_module
            rep[..., i] = code.get_rep(stim[:, sd_b:sd_e], add_noise=False)
            mask = np.all(np.isnan(stim[:, sd_b:sd_e]), axis=1, keepdims=True)
            rep[mask[:, 0], :, i] = 0
            off += mask
        rep = combine(rep, axis=2)
        scale = self.n_modules / (self.n_modules - off)
        rep = rep*np.sqrt(scale)
        if add_noise:
            rep = rep + self.noise.rvs(rep.shape)
        return rep

    def get_predicted_mse(self, code_ind=0, rate=None):
        if rate is not None:
            sigma = self._rate_sigma(rate)
            pwr = self._rate_pwr(rate)
        return self.code_list[code_ind].get_predicted_mse(pwr=pwr, sigma_n=sigma)

    def get_predicted_fi(self, code_ind=0, rate=None):
        if rate is not None:
            sigma = self._rate_sigma(rate)
            pwr = self._rate_pwr(rate)
        return self.code_list[code_ind].get_predicted_fi(pwr=pwr, sigma_n=sigma)


def estimate_effective_noise(code, single_dim=0, n_samps=10000):
    stim = code.sample_stim(n_samps)
    stim[:, single_dim] = .5
    
    reps = code.get_rep(stim)
    return np.mean(np.std(reps, axis=0))

    
def optimize_sigma_w(
    wid, pwr, n_units, n_modules, dims=1, sigma_n=1, delt=1e-5, max_iter=10, **kwargs
):
    if wid is None:
        wid = Code(pwr, n_units, dims=dims, wid=wid, sigma_n=sigma_n**2, **kwargs).wid

    add_sigma = (n_modules - 1) * rfm.random_uniform_unit_var(pwr, n_units, wid, dims)
    sigma = np.sqrt(sigma_n**2 + add_sigma)
    new_wid = Code(pwr, n_units, dims=dims, sigma_n=sigma, **kwargs).wid
    iter_ = 0
    while np.abs(wid - new_wid) > delt:
        # print(np.abs(wid - new_wid))
        iter_ += 1
        if iter_ > max_iter:
            print("exceeded maximum iterations")
            break
        wid = new_wid
        add_sigma = (n_modules - 1) * rfm.random_uniform_unit_var(
            pwr, n_units, wid, dims
        )
        sigma = np.sqrt(sigma_n**2 + add_sigma)
        new_wid = Code(pwr, n_units, dims=dims, sigma_n=sigma, **kwargs).wid
    return sigma, new_wid


class SuperposCode(MultiCode):
    def __init__(
            self,
            n_modules,
            pwr,
            n_units,
            dims=1,
            sigma_n=1,
            stim_distr=None,
            use_linear_transform=False,
            **kwargs,
    ):
        mod_pwr = pwr / n_modules
        self.mod_pwr = mod_pwr
        code_list = []
        wid_i = kwargs.get("wid")
        if wid_i is None:
            new_sigma, wid_i = optimize_sigma_w(
                wid_i,
                mod_pwr,
                n_units,
                n_modules=n_modules,
                dims=dims,
                sigma_n=sigma_n,
                **kwargs
            )
        all_cents = []
        for i in range(n_modules):
            code_i = Code(
                mod_pwr, n_units, dims=dims, wid=wid_i, sigma_n=new_sigma, **kwargs
            )
            all_cents.append(code_i.rf_cents)
            code_list.append(code_i)
        self.sigma_est = new_sigma
        self.rf_cents = np.concatenate(all_cents, 1)
        self.code_list = code_list
        self.dims_per_module = dims
        self.dims = dims * n_modules
        self.n_modules = n_modules
        self.n_units = n_units
        self.n_units_per_module = n_units
        self.noise = sts.norm(0, sigma_n)
        self.sigma_n = sigma_n
        self.linear_decoder = None
        self.wid = wid_i
        mean_interference = rfm.random_uniform_unit_mean(
            wid_i, dims, self.code_list[0].rf_scale
        )
        self.add_dc = (n_modules - 1) * mean_interference
        self.stim_distr = stim_distr
        if use_linear_transform:
            self.linear_transform = make_orthonormal_matrix(n_units)
        else:
            self.linear_transform = None

    def get_rep(self, *args, combine=np.nansum, **kwargs):
        return super().get_rep(*args, combine=combine, **kwargs)


class RescalingSuperposCode(RescalingMultiCode):
    def __init__(
            self,
            n_modules,
            pwr,
            n_units,
            dims=1,
            sigma_n=1,
            set_rate=None,
            **kwargs,
    ):
        if set_rate is None:
            set_rate = n_modules
        mod_pwr = pwr / set_rate
        code_list = []
        wid_i = kwargs.get("wid")
        if wid_i is None:
            new_sigma, wid_i = optimize_sigma_w(
                wid_i,
                mod_pwr,
                n_units,
                n_modules=set_rate,
                dims=dims,
                sigma_n=sigma_n,
                **kwargs
            )
        all_cents = []
        for i in range(n_modules):
            code_i = Code(
                mod_pwr, n_units, dims=dims, wid=wid_i, sigma_n=new_sigma, **kwargs
            )
            all_cents.append(code_i.rf_cents)
            code_list.append(code_i)
        self.sigma_est = new_sigma
        self.rf_cents = np.concatenate(all_cents, 1)
        self.code_list = code_list
        self.dims_per_module = dims
        self.dims = dims * n_modules
        self.n_modules = n_modules
        self.n_units = n_units
        self.n_units_per_module = n_units
        self.noise = sts.norm(0, sigma_n)
        self.sigma_n = sigma_n
        self.linear_decoder = None
        self.wid = wid_i
        self.pwr = pwr
        mean_interference = rfm.random_uniform_unit_mean(
            wid_i, dims, self.code_list[0].rf_scale
        )
        self.add_dc = (n_modules - 1) * mean_interference

    def _rate_pwr(self, rate):
        mod_pwr = self.pwr / rate
        return mod_pwr

    def _rate_sigma(self, rate):
        mod_pwr = self.pwr / rate
        add_sigma = (rate - 1) * rfm.random_uniform_unit_var(
            mod_pwr, self.n_units, self.wid, self.dims_per_module
        )
        sigma = np.sqrt(self.sigma_n**2 + add_sigma)
        return sigma

    def get_rep(self, *args, combine=np.nansum, **kwargs):
        return super().get_rep(*args, combine=combine, **kwargs)


def mod_concat(arr, axis=2, conc_ax=1):
    arr_m = np.moveaxis(arr, axis, 0)
    return np.concatenate(arr_m, axis=conc_ax)


class ModularCode(MultiCode):
    def __init__(
            self,
            n_modules,
            pwr,
            n_units,
            dims=1,
            sigma_n=1,
            stim_distr=None,
            use_linear_transform=False,
            **kwargs,
    ):
        mod_pwr = pwr / n_modules
        mod_units = np.round(n_units / n_modules).astype(int)
        wid_i = kwargs.get("wid")
        code_list = []
        all_cents = []
        module_members = np.zeros(n_units)
        module_members[:] = np.nan
        for i in range(n_modules):
            code_i = Code(
                mod_pwr, mod_units, dims=dims, wid=wid_i, sigma_n=sigma_n, **kwargs
            )
            wid_i = code_i.wid
            cents_i = [np.zeros((mod_units, dims))]*n_modules
            cents_i[i] = code_i.rf_cents
            module_members[i*mod_units:(i+1)*mod_units] = i
            all_cents.append(cents_i)
            code_list.append(code_i)
        self.module_members = module_members
        self.wid = wid_i
        self.rf_cents = np.block(all_cents)
        self.code_list = code_list
        self.dims_per_module = dims
        self.dims = dims * n_modules
        self.n_units_per_module = mod_units
        self.sigma_n = sigma_n
        self.n_units = n_units
        self.n_modules = n_modules
        self.noise = sts.norm(0, sigma_n)
        self.linear_decoder = None
        self.stim_distr = None
        self.add_dc = 0
        if use_linear_transform:
            self.linear_transform = make_orthonormal_matrix(n_units)
        else:
            self.linear_transform = None

    def get_rep(self, *args, combine=mod_concat, **kwargs):
        return super().get_rep(*args, combine=combine, **kwargs)


class RescalingModularCode(RescalingMultiCode):
    def __init__(
            self,
            n_modules,
            pwr,
            n_units,
            dims=1,
            sigma_n=1,
            set_rate=None,
            **kwargs,
    ):
        mod_pwr = pwr / n_modules
        mod_units = np.round(n_units / n_modules).astype(int)
        wid_i = kwargs.get("wid")
        code_list = []
        all_cents = []
        for i in range(n_modules):
            code_i = Code(
                mod_pwr, mod_units, dims=dims, wid=wid_i, sigma_n=sigma_n, **kwargs
            )
            wid_i = code_i.wid
            cents_i = [np.zeros((mod_units, dims))]*n_modules
            cents_i[i] = code_i.rf_cents
            all_cents.append(cents_i)
            code_list.append(code_i)
        self.wid = wid_i
        self.pwr = pwr
        self.rf_cents = np.block(all_cents)
        self.code_list = code_list
        self.dims_per_module = dims
        self.dims = dims * n_modules
        self.n_units_per_module = mod_units
        self.sigma_n = sigma_n
        self.n_units = n_units
        self.n_modules = n_modules
        self.noise = sts.norm(0, sigma_n)
        self.linear_decoder = None
        self.add_dc = 0

    def _rate_pwr(self, rate):
        mod_pwr = self.pwr / rate
        return mod_pwr

    def _rate_sigma(self, rate):
        return self.sigma_n

    def get_rep(self, *args, combine=mod_concat, **kwargs):
        return super().get_rep(*args, combine=combine, **kwargs)


@u.arg_list_decorator
def sweep_code_performance(
    pwrs, n_units, dims, n_samps=1000, code_type=Code, n_jobs=-1, n_cand=100, **kwargs
):
    mse_emp = np.zeros((len(pwrs), len(n_units), len(dims), n_samps))
    mse_boot = np.zeros_like(mse_emp)
    mse_theor = np.zeros(mse_emp.shape[:-1])
    fi_theor = np.zeros_like(mse_theor)

    def _sweep_helper(ind):
        p_i, nu_i, dim_i = ind
        code = code_type(pwrs[p_i], n_units[nu_i], dims=dims[dim_i], **kwargs)
        mse_theor_ind = code.get_predicted_mse()
        fi_theor_ind = 1 / code.get_predicted_fi()
        mse_out_ind = code.empirical_mse(n_samps=n_samps, n_candidates=n_cand)[:, 0]
        mse_boot_ind = u.bootstrap_list(mse_out_ind, np.mean, n=n_samps)
        return ind, mse_theor_ind, fi_theor_ind, mse_out_ind, mse_boot_ind

    ind_iter = u.make_array_ind_iterator(mse_theor.shape)
    par = jl.Parallel(n_jobs=n_jobs)
    out = par(jl.delayed(_sweep_helper)(ind) for ind in ind_iter)
    for ind, mse_theor_ind, fi_theor_ind, mse_out_ind, mse_boot_ind in out:
        mse_emp[ind] = mse_out_ind
        mse_boot[ind] = mse_boot_ind
        mse_theor[ind] = mse_theor_ind
        fi_theor[ind] = fi_theor_ind
    return mse_emp, mse_boot, mse_theor, fi_theor
