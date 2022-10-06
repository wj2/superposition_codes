
import os
import re
import pickle
import numpy as np

import general.utility as u

def load_sweeps(folder, jobid, template='code_s_sweep_[0-9]+_{jobid}\.pkl',
                params_key='params'):
    fls = os.listdir(folder)
    s_temp = template.format(jobid=jobid)
    pwr_emp_list = []
    nus_emp_list = []
    for fl in fls:
        m = re.match(s_temp, fl)
        if m is not None:
            fname = os.path.join(folder, fl)
            data = pickle.load(open(fname, 'rb'))
            pwrs, nus, dims = data[params_key]
            pwr_theor = data['pwr_sweep'][2]
            nus_theor = data['nu_sweep'][2]
            pwr_fi = data['pwr_sweep'][3]
            nus_fi = data['nu_sweep'][3]

            pwr_emp_list.append(data['pwr_sweep'][0])
            nus_emp_list.append(data['nu_sweep'][0])
            
    pwr_emp = np.concatenate(pwr_emp_list, axis=3)
    pwr_boot = np.zeros_like(pwr_emp)
    for ind in u.make_array_ind_iterator(pwr_emp.shape[:-1]):
        pwr_boot[ind] = u.bootstrap_list(pwr_emp[ind], np.mean,
                                         n=pwr_emp[ind].shape[0])
    nus_emp = np.concatenate(nus_emp_list, axis=3)
    nus_boot = np.zeros_like(nus_emp)
    for ind in u.make_array_ind_iterator(nus_emp.shape[:-1]):
        nus_boot[ind] = u.bootstrap_list(nus_emp[ind], np.mean,
                                         n=nus_emp[ind].shape[0])
    out = {'params':(pwrs, nus, dims),
           'args':data['args'],
           'pwr_sweep':(pwr_emp, pwr_boot, pwr_theor, pwr_fi),
           'nus_sweep':(nus_emp, nus_boot, nus_theor, nus_fi)}
    return out
