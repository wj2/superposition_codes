
import numpy as np
import scipy.stats as sts
import scipy.special as ss
import sklearn.linear_model as sklm
import joblib as jl

import general.rf_models as rfm
import general.utility as u

class Code:

    def __init__(self, pwr, n_units, dims=1, wid=None, lam=2,
                 spk_cost='l2', pop_func='random', sigma_n=1, modules=1):
        if spk_cost == 'l1':
            cost_func = lambda x: np.mean(np.sum(np.abs(x), axis=1))
            titrate_func = lambda x, y: x/y
        elif spk_cost == 'l2':
            cost_func = lambda x: np.mean(np.sum(x**2, axis=1))
            titrate_func = lambda x, y: np.sqrt(x/y)
        else:
            raise IOError('unrecognized spk_cost indicator {}'.format(spk_cost))
        if pop_func == 'random':
            pop_func = rfm.get_random_uniform_pop
        elif pop_func == 'lattice':
            pop_func = rfm.get_lattice_uniform_pop
        else:
            raise IOError('unrecognized pop_func indicator {}'.format(pop_func))

        if wid is None:
            out = rfm.min_mse_vec(pwr, n_units, dims, ret_components=True,
                                  sigma_n=sigma_n, lam=lam)
            wid = out[-1]
        
        pwr_pre = rfm.random_uniform_pwr(n_units, wid, dims, scale=1)
        rescale = np.sqrt(pwr/pwr_pre)
        out = pop_func(pwr, n_units, dims, w_use=wid,
                       sigma_n=sigma_n,
                       cost_func=cost_func,
                       titrate_func=titrate_func,
                       scale_use=rescale,
                       ret_params=True)
        stim_distr, rf, drf, noise, ms, ws = out
        
        self.rf_cents = ms
        self.rf_scale = rescale
        self.stim_distr = stim_distr
        self.rf = rf
        self.drf = drf
        self.noise = noise

        self.n_units = n_units
        self.pwr = pwr
        self.dims = dims
        self.wid = wid
        self.sigma_n = sigma_n

        self.linear_decoder = None
        self.add_dc = 0

    def get_predicted_fi(self):
        return rfm.random_uniform_fi_pwr(self.n_units, self.pwr, self.wid,
                                         self.dims, sigma_n=self.sigma_n)

    def get_predicted_threshold_err(self):
        pwr_var = self.get_activity_var()

        p, mag =  rfm.compute_threshold_vec(self.pwr, self.n_units, self.dims,
                                            self.wid)
        return p, mag

    def get_predicted_mse(self):
        fi = self.get_predicted_fi()
        t_prob, t_mag = self.get_predicted_threshold_err()
        return t_prob*t_mag + (1 - t_prob)*(1/fi)

    def get_activity_var(self):
        pwr_var = rfm.random_uniform_pwr_var_fix(self.n_units,
                                                 self.pwr,
                                                 self.wid,
                                                 self.dims)
        return pwr_var

    def get_predicted_fi_var(self):
        return rfm.random_uniform_fi_var_pwr(self.n_units,
                                             self.pwr,
                                             self.wid,
                                             self.dims)
    
    def generate_errors(self, n_samps=1000, eps=.0001):
        p = self.get_predicted_threshold_err()[0]
        fi = self.get_predicted_fi()
        fi_var = self.get_predicted_fi_var()
        rng = np.random.default_rng()
        fis = rng.normal(fi, np.sqrt(fi_var), size=(n_samps,))
        fis[fis < 0] = np.nan
        
        thr_err = np.abs(rng.uniform(0, 1, size=n_samps)
                         - rng.uniform(0, 1, size=n_samps))
        trl_type = rng.uniform(0, 1, size=n_samps) < p
        out = np.zeros(n_samps)
        out[trl_type] = thr_err[trl_type]
        out[np.logical_not(trl_type)] = 1/fis[np.logical_not(trl_type)]
        return out

    def sample_stim(self, n_stim=1000):
        return self.stim_distr.rvs(n_stim)
    
    def sample_reps(self, n_reps=1000, add_noise=True):
        stim = self.sample_stim(n_reps)
        reps = self.get_rep(stim, add_noise=add_noise)
        return stim, reps
        
    def get_rep(self, stim, add_noise=True):
        rep = self.rf(stim)
        if add_noise:
            rep = rep + self.noise.rvs(stim.shape[0])
        return rep

    def decode_rep_brute(self, rep, n_candidates=500):
        return rfm.brute_decode_rf(rep, self.get_rep, self.dims,
                                   n_gran=n_candidates, add_noise=False)

    def decode_rep_refine(self, rep, n_candidates=10):
        return rfm.refine_decode_rf(rep, self.get_rep, self.dims,
                                    n_gran=n_candidates, add_noise=False)

    def decode_rep_brute_decoupled(self, rep, dim_i=(0,), n_candidates=1000):
        return rfm.brute_decode_decouple(rep, self.get_rep, self.dims,
                                         dim_i=dim_i,
                                         n_gran=n_candidates,
                                         add_noise=False,
                                         add_dc=self.add_dc)

    def train_linear_decoder(self, model=sklm.Ridge, n_training=1000, **kwargs):
        m = model(**kwargs)
        stim, reps = self.sample_reps(n_training)
        m.fit(reps, stim)
        return m
    
    def decode_rep_linear(self, rep, n_training=1000, **kwargs):
        if self.linear_decoder is None:
            self.linear_decoder = self.train_linear_decoder(**kwargs)
        return self.linear_decoder.predict(rep)
    
    def decode_rep(self, rep, method='brute', **kwargs):
        if method == 'brute':
            out = self.decode_rep_brute(rep, **kwargs)
        elif method == 'brute_decoupled':
            out = self.decode_rep_brute_decoupled(rep, **kwargs)
        elif method == 'linear':
            out = self.decode_rep_linear(rep, **kwargs)
        elif method == 'refine':
            out = self.decode_rep_refine(rep, **kwargs)
        else:
            raise IOError('unrecognized decoding method')
        return out 
        
    def empirical_mse(self, n_samps=2000, boot=False, **kwargs):
        stim, rep = self.sample_reps(n_samps)
        stim_hat = self.decode_rep(rep, **kwargs)

        mse = (stim - stim_hat)**2
        if boot:
            mse = u.bootstrap_list(mse, np.mean, n=n_samps)
        return mse

class MultiCode(Code):

    def sample_stim(self, n_stim=1000):
        out = np.zeros((n_stim, self.dims))
        for i, code in enumerate(self.code_list):
            sd_b = i*self.dims_per_module
            sd_e = (i + 1)*self.dims_per_module
            out[:, sd_b:sd_e] = code.sample_stim(n_stim)
        return out

    def get_rep(self, stim, add_noise=True, combine=np.nansum):
        rep = np.zeros((stim.shape[0], self.n_units_per_module,
                        self.n_modules))
        for i, code in enumerate(self.code_list):
            sd_b = i*self.dims_per_module
            sd_e = (i + 1)*self.dims_per_module
            rep[..., i] = code.get_rep(stim[:, sd_b:sd_e],
                                       add_noise=False)
        rep = combine(rep, axis=2)
        if add_noise:
            rep = rep + self.noise.rvs(stim.shape[0])
        return rep

    def decode_rep(self, rep, method='brute_decoupled', **kwargs):
        dim_i = np.arange(self.dims_per_module, dtype=int)
        return super().decode_rep(rep, method=method, dim_i=dim_i,
                                  **kwargs)

    def get_predicted_mse(self, code_ind=0):
        return self.code_list[code_ind].get_predicted_mse()
    
    def get_predicted_fi(self, code_ind=0):
        return self.code_list[code_ind].get_predicted_fi()

def optimize_sigma_w(wid, pwr, n_units, n_modules, dims=1, sigma_n=1,
                     delt=1/10000, max_iter=10, **kwargs):
    if wid is None:
        wid = Code(pwr, n_units, dims=dims, wid=wid,
                   sigma_n=sigma_n**2, **kwargs).wid
        
    add_sigma = (n_modules - 1)*rfm.random_uniform_unit_var(pwr, n_units, wid,
                                                            dims)
    sigma = np.sqrt(sigma_n**2 + add_sigma)
    new_wid = Code(pwr, n_units, dims=dims,
                   sigma_n=sigma, **kwargs).wid
    iter_ = 0
    while np.abs(wid - new_wid) > delt:
        # print(np.abs(wid - new_wid))
        iter_ += 1
        if iter_ > max_iter:
            print('exceeded maximum iterations')
            break
        wid = new_wid
        add_sigma = (n_modules - 1)*rfm.random_uniform_unit_var(pwr, n_units,
                                                                wid, dims)
        sigma = np.sqrt(sigma_n**2 + add_sigma)
        new_wid = Code(pwr, n_units, dims=dims,
                       sigma_n=sigma, **kwargs).wid
    return sigma, wid
    
class SuperposCode(MultiCode):

    def __init__(self, n_modules, pwr, n_units, dims=1, sigma_n=1, **kwargs):
        mod_pwr = pwr/n_modules
        code_list = []
        wid_i = kwargs.get('wid')
        if wid_i is None:
            new_sigma, wid_i = optimize_sigma_w(wid_i, mod_pwr, n_units,
                                                n_modules, dims,
                                                sigma_n, **kwargs)
        for i in range(n_modules):
            code_i = Code(mod_pwr, n_units, dims=dims, wid=wid_i,
                          sigma_n=new_sigma, **kwargs)
            code_list.append(code_i)
        self.code_list = code_list
        self.dims_per_module = dims
        self.dims = dims*n_modules
        self.n_modules = n_modules
        self.n_units = n_units
        self.n_units_per_module = n_units
        self.noise = sts.multivariate_normal(np.zeros(n_units), sigma_n)
        self.linear_decoder = None
        mean_interference = rfm.random_uniform_unit_mean(
            wid_i, dims, self.code_list[0].rf_scale)
        self.add_dc = (n_modules - 1)*mean_interference

    def get_rep(self, *args, combine=np.nansum, **kwargs):
        return super().get_rep(*args, combine=combine, **kwargs)

def mod_concat(arr, axis=2, conc_ax=1):
    arr_m = np.moveaxis(arr, axis, 0)
    return np.concatenate(arr_m, axis=conc_ax)
        
class ModularCode(MultiCode):

    def __init__(self, n_modules, pwr, n_units, dims=1, sigma_n=1,
                 **kwargs):
        mod_pwr = pwr/n_modules
        mod_units = np.round(n_units/n_modules).astype(int)
        wid_i = kwargs.get('wid')
        code_list = []
        for i in range(n_modules):
            code_i = Code(mod_pwr, mod_units, dims=dims, wid=wid_i,
                          sigma_n=sigma_n, **kwargs)
            wid_i = code_i.wid
            code_list.append(code_i)
        self.code_list = code_list
        self.dims_per_module = dims
        self.dims = dims*n_modules
        self.n_units_per_module = mod_units
        self.n_units = n_units
        self.n_modules = n_modules
        self.noise = sts.multivariate_normal(np.zeros(mod_units*n_modules),
                                             sigma_n)
        self.linear_decoder = None
        self.add_dc = 0

    def get_rep(self, *args, combine=mod_concat, **kwargs):
        return super().get_rep(*args, combine=combine, **kwargs)


@u.arg_list_decorator
def sweep_code_performance(pwrs, n_units, dims, n_samps=1000, code_type=Code,
                           n_jobs=-1, n_cand=100, **kwargs):
    mse_emp = np.zeros((len(pwrs), len(n_units), len(dims), n_samps))
    mse_boot = np.zeros_like(mse_emp)
    mse_theor = np.zeros(mse_emp.shape[:-1])
    fi_theor = np.zeros_like(mse_theor)

    def _sweep_helper(ind):
        p_i, nu_i, dim_i = ind
        code = code_type(pwrs[p_i], n_units[nu_i], dims=dims[dim_i], **kwargs)
        mse_theor_ind = code.get_predicted_mse()
        fi_theor_ind = 1/code.get_predicted_fi()
        mse_out_ind = code.empirical_mse(n_samps=n_samps,
                                         n_candidates=n_cand)[:, 0]
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
