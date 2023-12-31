from datetime import datetime
from locale import normalize
import numpy as np
import argparse
import time
import pandas as pd
import pickle

from var_objective.differential_operator import LinearOperator
from var_objective.utils.gp_utils import gp_to_pysym_no_coef, gp_to_pysym_with_coef
from var_objective.utils.logging import create_logging_file_names, create_output_dir

from .equations import get_pdes
from .grids import EquiPartGrid
from .generator import generate_fields
from .interpolate import estimate_fields
from .basis import BSplineFreq2D, BSplineTrans1D, FourierSine2D, BSplineTrans2D
from .optimize_operator import VariationalWeightsFinder, VariationalWeightsFinderDictionary
from .conditions import get_conditions_set
from .config import get_optim_params, get_gp_params
from .libs import SymbolicRegressor, make_fitness

import sympy

import pysindy as ps
from pysindy.differentiation import SmoothedFiniteDifference

def grid_and_fields_to_covariates(grid_and_fields):

    grid_and_fields = np.moveaxis(grid_and_fields,1,-1)
    num_var = grid_and_fields.shape[-1]
    return np.reshape(grid_and_fields,(-1,num_var))


def _check_if_zero(vector):
    if np.sum(vector == 0.0) == len(vector):
        return True
    else:
        return False

def save_output(filename, trial, seed, target_weights, best_found_weights, pde_find_target_weights, 
pde_find_stlsq_best_found_weights,
pde_find_frols_best_found_weights,
pde_find_sr3_best_found_weights,
pde_find_srr_best_found_weights,
weak_sindy_stlsq_best_found_weights,
weak_sindy_frols_best_found_weights,
weak_sindy_sr3_best_found_weights,
weak_sindy_srr_best_found_weights,
var_error, 
pde_find_stlsq_error,
pde_find_frols_error,
pde_find_sr3_error,
pde_find_srr_error,
weak_sindy_stlsq_error,
weak_sindy_frols_error,
weak_sindy_sr3_error,
weak_sindy_srr_error,
time_elapsed):
    message = f"""
----------------------
Trial: {trial}
Seed: {seed}
Var_target_weights: {target_weights}
Var_best_found_weights: {best_found_weights}
PDE-FIND_target_weights: {pde_find_target_weights}
PDE-FIND_stlsq_best_found_weights: {pde_find_stlsq_best_found_weights}
PDE-FIND_frols_best_found_weights: {pde_find_frols_best_found_weights}
PDE-FIND_sr3_best_found_weights: {pde_find_sr3_best_found_weights}
PDE-FIND_srr_best_found_weights: {pde_find_srr_best_found_weights}
WeakSindy_stlsq_best_found_weight: {weak_sindy_stlsq_best_found_weights}
WeakSindy_frols_best_found_weight: {weak_sindy_frols_best_found_weights}
WeakSindy_sr3_best_found_weight: {weak_sindy_sr3_best_found_weights}
WeakSindy_srr_best_found_weight: {weak_sindy_srr_best_found_weights}
Var_error: {var_error}
PDE-FIND_stlsq_error: {pde_find_stlsq_error}
PDE-FIND_frols_error: {pde_find_frols_error}
PDE-FIND_sr3_error: {pde_find_sr3_error}
PDE-FIND_srr_error: {pde_find_srr_error}
WeakSindy_stlsq_error: {weak_sindy_stlsq_error}
WeakSindy_frols_error: {weak_sindy_frols_error}
WeakSindy_sr3_error: {weak_sindy_sr3_error}
WeakSindy_srr_error: {weak_sindy_srr_error}
Time elapsed: {time_elapsed}"""

    with open(filename, 'a') as f:
        f.write(message)

def df_append(old_df, trial, seed, target_weights, best_found_weights, pde_find_target_weights, 
pde_find_stlsq_best_found_weights,
pde_find_frols_best_found_weights,
pde_find_sr3_best_found_weights,
pde_find_srr_best_found_weights,
weak_sindy_stlsq_best_found_weights,
weak_sindy_frols_best_found_weights,
weak_sindy_sr3_best_found_weights,
weak_sindy_srr_best_found_weights,
var_error, 
pde_find_stlsq_error,
pde_find_frols_error,
pde_find_sr3_error,
pde_find_srr_error,
weak_sindy_stlsq_error,
weak_sindy_frols_error,
weak_sindy_sr3_error,
weak_sindy_srr_error,
time_elapsed):
    df = pd.DataFrame()
    df['trial'] = [trial]
    df['seed'] = [seed]
    for i,x in enumerate(target_weights):
        df[f'var_target_weights_{i}'] = [x]
    for i,x in enumerate(pde_find_target_weights):
        df[f'pde_find_target_weights_{i}'] = [x]
    for i,x in enumerate(best_found_weights):
        df[f'var_weights_{i}'] = [x]
    for i,x in enumerate(pde_find_stlsq_best_found_weights):
        df[f'pde_find_stlsq_weights_{i}'] = [x]
    for i,x in enumerate(pde_find_frols_best_found_weights):
        df[f'pde_find_frols_weights_{i}'] = [x]
    for i,x in enumerate(pde_find_sr3_best_found_weights):
        df[f'pde_find_sr3_weights_{i}'] = [x]
    for i,x in enumerate(pde_find_srr_best_found_weights):
        df[f'pde_find_srr_weights_{i}'] = [x]
    for i,x in enumerate(weak_sindy_stlsq_best_found_weights):
        df[f'weak_sindy_stlsq_weights_{i}'] = [x]
    for i,x in enumerate(weak_sindy_frols_best_found_weights):
        df[f'weak_sindy_frols_weights_{i}'] = [x]
    for i,x in enumerate(weak_sindy_sr3_best_found_weights):
        df[f'weak_sindy_sr3_weights_{i}'] = [x]
    for i,x in enumerate(weak_sindy_srr_best_found_weights):
        df[f'weak_sindy_srr_weights_{i}'] = [x]
    df['var_error'] = [var_error]
    df['pde_find_stlsq_error'] = [pde_find_stlsq_error]
    df['pde_find_frols_error'] = [pde_find_frols_error]
    df['pde_find_sr3_error'] = [pde_find_sr3_error]
    df['pde_find_srr_error'] = [pde_find_srr_error]
    df['weak_sindy_stlsq_error'] = [weak_sindy_stlsq_error]
    df['weak_sindy_frols_error'] = [weak_sindy_frols_error]
    df['weak_sindy_sr3_error'] = [weak_sindy_sr3_error]
    df['weak_sindy_srr_error'] = [weak_sindy_srr_error]
  
    df['time_elapsed'] = [time_elapsed]

    df_new = pd.concat([old_df,df],axis=0)
    return df_new

def save_meta(filename_pickle, filename_df, global_seed, arguments, gp_config):
    meta = {'global_seed':global_seed,'arguments':arguments,'gp_config':gp_config,'table':filename_df}
    with open(filename_pickle, 'wb') as file:
        pickle.dump(meta,file)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Discover a PDE")
    parser.add_argument('name', help='Equation name from equations.py')
    parser.add_argument('equation_number', type=int, help='Which equation to discover?')
    parser.add_argument('width', type=float, help='Width of the grid')
    parser.add_argument('frequency_per_dim', type=int, help='Frequency per dimension of generated data')
    parser.add_argument('noise_ratio', type=float, help='Noise ration for data generation')
    parser.add_argument('full_grid_samples', type=int, help='Frequency of the full grid')
    parser.add_argument('conditions_set', help='Conditions set name from conditions.py')
    parser.add_argument('basis', choices=['fourier','2spline2D', '2spline2Dtrans', '2spline1Dtrans', '4spline2Dtrans'])
    parser.add_argument('max_ind_basis', type=int, help='Maximum index for test functions. Number of used test functions is a square of this number')
    parser.add_argument('num_trials', type=int, help='Number of trials')
    parser.add_argument('normalization',choices=['l1','l2'])
    parser.add_argument('solver', help='Least squares solver')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--sign_index', type=int, default=1)
    parser.add_argument('--sindy_order', type=int, default=2)
    parser.add_argument('--exp_name', help='Experiment name', default='new_experiment')
    parser.add_argument('--generations', type=int, default=20)

    args = parser.parse_args()

    INF_FLOAT = 9.0e+300
    LSTSQ_SOLVER = args.solver

    pdes = get_pdes(args.name)

    M = pdes.M

    widths = [args.width] * M

    observed_grid = EquiPartGrid(widths, args.frequency_per_dim)
    full_grid = EquiPartGrid(widths, args.full_grid_samples)

    output_dir = create_output_dir(args.exp_name, 'all')

    dt = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    filename, filename_csv, filename_meta = create_logging_file_names(output_dir, dt)

    gp_params = get_gp_params(generations=args.generations)

    gp_params = get_gp_params()

    with open(filename, 'w') as f:
        f.write(f"""
Global seed: {args.seed}
Parameters: {args}
gplearn config: {gp_params}
        """)
    
    if args.num_trials == 0:
        seeds = [args.seed]
    else:
        np.random.seed(args.seed)
        seeds = np.random.randint(0,1000000,size=args.num_trials)

    df = pd.DataFrame()
    save_meta(filename_meta,filename_csv,args.seed,args,gp_params)

    for trial, seed in enumerate(seeds):

        trial_start = time.time()

        print(f'Trial {trial+1}/{len(seeds)}')

        conditions = get_conditions_set(args.conditions_set, params={'seed': seed, 'num_samples':args.num_samples})

        print(f"Seed set to {seed}")
        print(f"Generating dataset of {args.name} on a grid with width {args.width}, frequency per dim {args.frequency_per_dim}, noise ratio {args.noise_ratio} and using conditions set {args.conditions_set}")
        start = time.time()
        observed_dataset = generate_fields(pdes, conditions, observed_grid, args.noise_ratio, seed=seed)
        end = time.time()
        print(f"Observed dataset generated in {end-start} seconds")

        print(f"Estimating fields on a grid with frequency {args.full_grid_samples} per dimension")
        start = time.time()
        full_dataset = estimate_fields(observed_grid,observed_dataset,full_grid,seed=seed)
        end = time.time()
        print(f"Fields estimated in {end-start} seconds")

        dictQ = pdes.get_dictionaries()[args.equation_number]

        if args.basis == 'fourier':
            basis = FourierSine2D(widths)
            index_limits = [args.max_ind_basis] * 2
        elif args.basis == '2spline2D':
            basis = BSplineFreq2D(widths, 2)
            index_limits = [args.max_ind_basis] * 2
        elif args.basis == '2spline2Dtrans':
            index_limits = [args.max_ind_basis] * 2
            basis = BSplineTrans2D(widths, 2, index_limits)
        elif args.basis == '2spline1Dtrans':
            index_limits = [args.max_ind_basis]
            basis = BSplineTrans1D(widths, 2, index_limits)
        elif args.basis == '3spline2Dtrans':
            index_limits = [args.max_ind_basis] * 2
            basis = BSplineTrans2D(widths, 3, index_limits)
        elif args.basis == '4spline2Dtrans':
            index_limits = [args.max_ind_basis] * 2
            basis = BSplineTrans2D(widths, 4, index_limits)

        print("Initializing Variational Weights Finder")
        start = time.time()
        var_wf = VariationalWeightsFinderDictionary(full_dataset,full_grid,dictQ,basis=basis,index_limits=index_limits,optim_engine=LSTSQ_SOLVER,seed=seed)
        end = time.time()
        print(f"Weight Finder initialized in {end-start} seconds")
        
        X = grid_and_fields_to_covariates(var_wf.grid_and_fields)
       
        target_weights = pdes.get_weights(normalize=True)[args.equation_number]
        g_target = pdes.get_free_parts(normalize=True)[args.equation_number]
        target_g_numpy = pdes.numpify_g(g_target)
        variables_part = [X[:,i] for i in range(pdes.M+pdes.N)]

        target_g_part = target_g_numpy(*variables_part)

        # Check if target_g_part is an array
        if not hasattr(target_g_part, "__len__"):
            # This means that it is a scalar
            target_g_part = float(target_g_part)
            if target_g_part == 0.0:
                target_g_part = None
            else:
                target_g_part = np.ones(X.shape[0]) * target_g_part
                # TODO: maybe in the future leverage the fact that it is a scalar
        
        target_loss = var_wf._calculate_loss(target_g_part, target_weights)
        # print(f"Loss with target weights and target g_part: {target_loss}")
        # print(f"Target weights: {target_weights}")

        best_found_loss, best_found_weights = var_wf.find_weights(target_g_part)
        # print(f"Loss for the best found weights: {best_found_loss}")
        # print(f"Best found weights: {best_found_weights}")

        if best_found_weights[args.sign_index] < 0:
            best_found_weights *= -1

        var_error = np.sqrt(np.mean((best_found_weights - target_weights) ** 2))
        # print(var_error)

        # SINDy methods

        t = observed_grid.axes[0]
        x = observed_grid.axes[1]

        u = np.swapaxes(observed_dataset[:,0,:,:],-1,-2)
        u = np.expand_dims(u,-1)
        u = [u[i] for i in range(u.shape[0])]

        dt = t[1]-t[0]

        library_functions = [lambda x: x]
        library_function_names = [lambda x: x]

        X, T = np.meshgrid(x, t)
        XT = np.asarray([X, T]).T

        threshold = 0.0


        def test(smoothing, optimizer, weak=False):
            if weak:
                pde_lib = ps.WeakPDELibrary(library_functions=library_functions,
                                    function_names=library_function_names,
                                    derivative_order=args.sindy_order,
                                    spatiotemporal_grid=XT,
                                    is_uniform=True, K=1000,
                                    )
            else:
                pde_lib = ps.PDELibrary(
                        library_functions=library_functions,
                        function_names=library_function_names,
                        derivative_order=args.sindy_order,
                        spatial_grid=x,
                        is_uniform=True,
                        )
            if smoothing == 'sfd':
                smo = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 5})
            elif smoothing == 'spline':
                smo = ps.SINDyDerivative(kind="spline", s=1e-2)
            elif smoothing == 'trend':
                smo = ps.SINDyDerivative(kind="trend_filtered", order=0, alpha=1e-2)
            
            alpha = 0.05
            
            print(f"Smo: {smoothing}, Opt: {optimizer}, alpha:{alpha}, weak:{weak}")
            if optimizer == 'stlsq':
                opt = ps.STLSQ(threshold=threshold, alpha=alpha)
            elif optimizer == 'frols':
                opt = ps.FROLS(alpha=alpha)
            elif optimizer == 'sr3':
                opt = ps.SR3(threshold=threshold)
            elif optimizer == 'srr':
                opt = ps.SSR(alpha=alpha)

            # print(type(smo))
            # print(type(pde_lib))
            if weak:
                model = ps.SINDy(feature_library=pde_lib, optimizer=opt)
                model.fit(u, multiple_trajectories=True)
            else:
                model = ps.SINDy(differentiation_method=smo,feature_library=pde_lib, optimizer=opt)
                model.fit(u, t=dt, multiple_trajectories=True)
        
            pde_find_weights = np.insert(np.array(model.coefficients()).reshape(-1),0,1.0)
            pde_find_weights /= np.linalg.norm(pde_find_weights,1)
            pde_find_target_weights = pdes.get_sindy_weights()[args.equation_number]
            pde_find_target_weights /= np.linalg.norm(pde_find_target_weights,1)
            # print(model.coefficients())
            pde_find_error = np.sqrt(np.mean((pde_find_weights - pde_find_target_weights) ** 2))
            # print(pde_find_error)
            return pde_find_error, pde_find_weights

        stlsq_strong_error, stlsq_strong_weights = test('sfd','stlsq',weak=False)
        frols_strong_error, frols_strong_weights = test('sfd','frols',weak=False)
        sr3_strong_error, sr3_strong_weights = test('sfd','sr3',weak=False)
        srr_strong_error, srr_strong_weights = test('sfd','srr',weak=False)
        stlsq_weak_error, stlsq_weak_weights = test('sfd','stlsq',weak=True)
        frols_weak_error, frols_weak_weights = test('sfd','frols',weak=True)
        sr3_weak_error, sr3_weak_weights = test('sfd','sr3',weak=True)
        srr_weak_error, srr_weak_weights = test('sfd','srr',weak=True)

        trial_end = time.time()

        pde_find_target_weights = pdes.get_sindy_weights()[args.equation_number]
        pde_find_target_weights /= np.linalg.norm(pde_find_target_weights,1)

        save_output(filename, trial+1, seed, target_weights, best_found_weights, pde_find_target_weights, 
        stlsq_strong_weights, frols_strong_weights, sr3_strong_weights, srr_strong_weights, stlsq_weak_weights, frols_weak_weights, sr3_weak_weights, srr_weak_weights,
        var_error, 
        stlsq_strong_error, frols_strong_error, sr3_strong_error, srr_strong_error, stlsq_weak_error, frols_weak_error, sr3_weak_error, srr_weak_error,
        trial_end-trial_start)
        
        df = df_append(df,trial+1, seed, target_weights, best_found_weights, pde_find_target_weights, 
        stlsq_strong_weights, frols_strong_weights, sr3_strong_weights, srr_strong_weights, stlsq_weak_weights, frols_weak_weights, sr3_weak_weights, srr_weak_weights,
        var_error, 
        stlsq_strong_error, frols_strong_error, sr3_strong_error, srr_strong_error, stlsq_weak_error, frols_weak_error, sr3_weak_error, srr_weak_error,
        trial_end-trial_start)

        df.to_csv(filename_csv)


       