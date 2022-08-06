from datetime import datetime
import numpy as np
import argparse
import time
import pandas as pd
import pickle

from var_objective.differential_operator import LinearOperator
from var_objective.utils.gp_utils import gp_to_pysym_no_coef, gp_to_pysym_with_coef

from .equations import get_pdes
from .grids import EquiPartGrid
from .generator import generate_fields
from .interpolate import estimate_fields
from .basis import BSplineFreq2D, BSplineTrans1D, FourierSine2D, BSplineTrans2D
from .optimize_operator import VariationalWeightsFinder, VariationalWeightsFinderDictionary
from .conditions import get_conditions_set
from .config import get_optim_params, get_gp_params
from .libs import SymbolicRegressor, make_fitness

import pysindy as ps
from pysindy.differentiation import SmoothedFiniteDifference
import sympy

def grid_and_fields_to_covariates(grid_and_fields):

    grid_and_fields = np.moveaxis(grid_and_fields,1,-1)
    num_var = grid_and_fields.shape[-1]
    return np.reshape(grid_and_fields,(-1,num_var))


def _check_if_zero(vector):
    if np.sum(vector == 0.0) == len(vector):
        return True
    else:
        return False

def save_output(filename, trial, seed, program, eqC, is_correct, raw_program, operator, loss, target_loss, target_loss_better_weights, target_weights, best_found_weights, time_elapsed):
    message = f"""
----------------------
Trial: {trial}
Seed: {seed}
Program: {program}
Operator: {operator}
Functional form: {eqC}
Is correct: {is_correct}
Loss: {loss}
Target_loss: {target_loss}
Target_weights: {target_weights}
Target_loss_better_weights: {target_loss_better_weights}
Best_found_weights: {best_found_weights}
Raw_program: {raw_program}
Time elapsed: {time_elapsed}"""
    with open(filename, 'a') as f:
        f.write(message)

def df_append(old_df, trial, seed, program, eqC, is_correct, raw_program, operator, loss, target_loss, target_loss_better_weights, target_weights, best_found_weights, time_elapsed):
    df = pd.DataFrame()
    df['trial'] = [trial]
    df['seed'] = [seed]
    df['program'] = [program]
    df['eqC'] = [eqC]
    df['is_correct'] = [f'{is_correct}']
    weights = operator.vectorize()[1:] #exclude 0 partial
    for i,x in enumerate(weights):
        df[f'operator_{i}'] = x
    df['loss'] = [loss]
    df['target_loss'] = [target_loss]
    df['target_loss_better_weights'] = [target_loss_better_weights]
    df['time_elapsed'] = [time_elapsed]
    df['raw_program'] = [raw_program]
    for i,x in enumerate(target_weights):
        df[f'target_weights_{i}'] = [x]
    for i,x in enumerate(best_found_weights):
        df[f'best_found_weights_{i}'] = [x]

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
    parser.add_argument('conditions_set', help='Conditions set name from conditions.py')
    parser.add_argument('num_trials', type=int, help='Number of trials')
    parser.add_argument('normalization',choices=['l1','l2'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)

    args = parser.parse_args()

    INF_FLOAT = 9.0e+300

    pdes = get_pdes(args.name)

    M = pdes.M

    widths = [args.width] * M

    observed_grid = EquiPartGrid(widths, args.frequency_per_dim)

    # dt = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    # filename = f"results/run_var_square_{dt}.txt"
    # filename_csv = f"results/run_var_square_{dt}_table.csv"
    # filename_meta = f"results/run_var_square_{dt}_meta.p"

#     gp_params = get_gp_params()

#     with open(filename, 'w') as f:
#         f.write(f"""
# Global seed: {args.seed}
# Parameters: {args}
# gplearn config: {gp_params}
#         """)
    
    if args.num_trials == 0:
        seeds = [args.seed]
    else:
        np.random.seed(args.seed)
        seeds = np.random.randint(0,1000000,size=args.num_trials)

    df = pd.DataFrame()
    # save_meta(filename_meta,filename_csv,args.seed,args,gp_params)

    for trial, seed in enumerate(seeds):

        print(f'Trial {trial+1}/{len(seeds)}')

        conditions = get_conditions_set(args.conditions_set, params={'seed': seed, 'num_samples':args.num_samples})

        print(f"Seed set to {seed}")
        print(f"Generating dataset of {args.name} on a grid with width {args.width}, frequency per dim {args.frequency_per_dim}, noise ratio {args.noise_ratio} and using conditions set {args.conditions_set}")
        start = time.time()
        observed_dataset = generate_fields(pdes, conditions, observed_grid, args.noise_ratio, seed=seed)
        end = time.time()
        print(f"Observed dataset generated in {end-start} seconds")
        
        t = observed_grid.axes[0]
        x = observed_grid.axes[1]

        u = np.swapaxes(observed_dataset[:,0,:,:],-1,-2)
        u = np.expand_dims(u,-1)
        u = [u[i] for i in range(u.shape[0])]

        dt = t[1]-t[0]

        library_functions = [lambda x: x]
        library_function_names = [lambda x: x]

        
        sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})

        pde_lib = ps.PDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            derivative_order=2,
            spatial_grid=x,
            is_uniform=True,
        )

       

        print('STLSQ model: ')
        optimizer = ps.STLSQ(threshold=0, alpha=0.05, normalize_columns=True)
        model = ps.SINDy(differentiation_method=sfd,feature_library=pde_lib, optimizer=optimizer)
        model.fit(u, t=dt, multiple_trajectories=True)
        print(pde_lib.get_feature_names())
        model.print()
        print(model.coefficients())

        X, T = np.meshgrid(x, t)
        XT = np.asarray([X, T]).T
        weak_pde_lib = ps.WeakPDELibrary(library_functions=library_functions,
                                    function_names=library_function_names,
                                    derivative_order=2,
                                    spatiotemporal_grid=XT,
                                    is_uniform=True, K=1000,
                                    )

        # Fit a weak form model
        # optimizer = ps.SR3(threshold=0.1, thresholder='l0',
        #                 tol=1e-8, normalize_columns=True, max_iter=1000)
        optimizer = ps.STLSQ(threshold=0, alpha=0.05, normalize_columns=True)
        model = ps.SINDy(feature_library=weak_pde_lib, optimizer=optimizer)
        model.fit(u, multiple_trajectories=True)
        print(weak_pde_lib.get_feature_names())
        model.print()


                

        
