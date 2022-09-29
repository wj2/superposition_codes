
import argparse
import pickle
import numpy as np
import superposition_codes.codes as spc


def create_parser():
    parser = argparse.ArgumentParser(description='fit several modularizers')
    parser.add_argument('-o', '--output_file',
                        default='assignment/code_param_sweep.pkl', type=str)
    parser.add_argument('--pwr_range', default=(.5, 2, 100), nargs=3,
                        type=float)
    parser.add_argument('--nu_range', default=(2, 3.2, 100), nargs=3,
                        type=float)
    parser.add_argument('--pwr_sweep_nu', default=1000, type=int)
    parser.add_argument('--nu_sweep_pwr', default=75)
    parser.add_argument('--probe_dims', default=(1, 2), nargs='+',
                        type=int)
    parser.add_argument('--n_samps', default=10, type=int)
    parser.add_argument('--code_type', default='code', type=str)
    parser.add_argument('--n_modules', default=1, type=int)
    return parser

if __name__ == '__main__':
    pwr_range = np.logspace(*args.pwr_range[:2], int(args.pwr_range[2]))
    nu_range = np.logspace(*args.nu_range[:2], int(args.nu_range[2]),
                           dtype=int)
    
    dims = args.prob_dims
    n_samps = args.n_samps

    parser = create_parser()
    args = parser.parse_args()

    fname = args.output_file
    if args.code_type == 'code':
        code_type = spc.Code
    elif args.code_type == 'modular':
        code_type = ft.partial(spc.ModularCode, n_modules)
    elif args.code_type == 'superposition':
        code_type = ft.partial(spc.SuperposCode, n_modules)
    else:
        raise IOError('unrecognized code type, {}'.format(args.code_type))
    
    out_pwr = spc.sweep_code_performance(pwr_range, args.pwr_sweep_nu, dims,
                                         n_samps=n_samps,
                                         code_type=code_type)
    out_nu = spc.sweep_code_performance(args.nu_sweep_pwr, nu_range, dims,
                                        n_samps=n_samps,
                                        code_type=code_type)
    out = {'params':(pwr_range, nu_range, dims), 'pwr_sweep_ind':pwr_fix_ind,
           'nus_sweep_ind':nus_fix_ind,
           'pwr_sweep':out_pwr, 'nu_sweep':out_nu}
    pickle.dump(out, open(fname, 'wb'))
    
