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

def save_output_no_gp(filename, trial, seed, target_weights, best_found_weights, var_error, time_elapsed):
    message = f"""
----------------------
Trial: {trial}
Seed: {seed}
Var_target_weights: {target_weights}
Var_best_found_weights: {best_found_weights}
Var_error: {var_error}
Time elapsed: {time_elapsed}"""

    with open(filename, 'a') as f:
        f.write(message)

def df_append_no_gp(old_df, trial, seed, target_weights, best_found_weights, var_error, time_elapsed):
    df = pd.DataFrame()
    df['trial'] = [trial]
    df['seed'] = [seed]
    for i,x in enumerate(target_weights):
        df[f'var_target_weights_{i}'] = [x]
    for i,x in enumerate(best_found_weights):
        df[f'var_weights_{i}'] = [x]
    df['var_error'] = [var_error]
    df['time_elapsed'] = [time_elapsed]

    df_new = pd.concat([old_df,df],axis=0)
    return df_new

def df_append(old_df, trial, seed, program, eqC, is_correct, raw_program, operator, loss, target_loss, target_loss_better_weights, target_weights, best_found_weights, time_elapsed):
    df = pd.DataFrame()
    df['trial'] = [trial]
    df['seed'] = [seed]
    df['program'] = [program]
    df['eqC'] = [eqC]
    df['is_correct'] = [f'{is_correct}']
    weights = operator
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
    parser.add_argument('full_grid_samples', type=int, help='Frequency of the full grid')
    parser.add_argument('conditions_set', help='Conditions set name from conditions.py')
    parser.add_argument('basis', choices=['fourier','2spline2D', '2spline2Dtrans', '2spline1Dtrans', '3spline2Dtrans'])
    parser.add_argument('max_ind_basis', type=int, help='Maximum index for test functions. Number of used test functions is a square of this number')
    parser.add_argument('num_trials', type=int, help='Number of trials')
    parser.add_argument('normalization',choices=['l1','l2'])
    parser.add_argument('solver', help='Least squares solver')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--no_gp', action='store_true', default=False)
    parser.add_argument('--sign_index', type=int, default=1)

    args = parser.parse_args()

    INF_FLOAT = 9.0e+300
    LSTSQ_SOLVER = args.solver

    pdes = get_pdes(args.name)

    M = pdes.M

    widths = [args.width] * M

    observed_grid = EquiPartGrid(widths, args.frequency_per_dim)
    full_grid = EquiPartGrid(widths, args.full_grid_samples)

    dt = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    filename = f"results/run_var_square_{dt}.txt"
    filename_csv = f"results/run_var_square_{dt}_table.csv"
    filename_meta = f"results/run_var_square_{dt}_meta.p"

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

        print("Initializing Variational Weights Finder")
        start = time.time()
        var_wf = VariationalWeightsFinderDictionary(full_dataset,full_grid,dictQ,basis=basis,index_limits=index_limits,optim_engine=LSTSQ_SOLVER,seed=seed)
        end = time.time()
        print(f"Weight Finder initialized in {end-start} seconds")


        def _var_fitness(y, y_pred, w):

            if len(y_pred) == 2:
                return 0.0

            if _check_if_zero(y_pred):
                loss, weights = var_wf.find_weights(None)
            else:
                loss, weights = var_wf.find_weights(y_pred)

            if loss is None:
                return INF_FLOAT

            return loss
        
        X = grid_and_fields_to_covariates(var_wf.grid_and_fields)
        fake_y = np.zeros(X.shape[0])

        var_fitness = make_fitness(_var_fitness, greater_is_better=False)
       
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
        
        target_loss = var_wf._calculate_loss(target_g_part, target_weights, dictQ=dictQ)
        print(f"Loss with target weights and target g_part: {target_loss}")
        print(f"Target weights: {target_weights}")

        best_found_loss, best_found_weights = var_wf.find_weights(target_g_part)
        print(f"Loss for the best found weights: {best_found_loss}")
        print(f"Best found weights: {best_found_weights}")

        if args.no_gp:

            if best_found_weights[args.sign_index] < 0:
                best_found_weights *= -1

            var_error = np.sqrt(np.mean((best_found_weights - target_weights) ** 2))

            trial_end = time.time()

            save_output_no_gp(filename, trial+1, seed, target_weights, best_found_weights, var_error, trial_end-trial_start)
            
            df = df_append_no_gp(df,trial+1, seed, target_weights, best_found_weights, var_error, trial_end-trial_start)

            df.to_csv(filename_csv)

        else:

            print(f"Starting evolution with population {gp_params['population_size']} and {gp_params['generations']} generations")
            start = time.time()
            est = SymbolicRegressor(metric=var_fitness, **gp_params ,verbose=1, random_state=seed)

            est.fit(X, fake_y)

            loss, weights = var_wf.find_weights(est.predict(X))

            print(weights)
            print(est._program)


            if weights[0] < 0:
                reverse = True
            else:
                reverse = False
            is_correct = "NA"
            try:
                eq, eqC = gp_to_pysym_with_coef(est)
                if reverse:
                    eq = (-1) * eq
                    eqC = (-1) * eqC
                true_g = pdes.get_functional_form_normalized(norm=args.normalization)[args.equation_number]
                expr = sympy.parsing.sympy_parser.parse_expr(f"{true_g} - ({eqC})")
                is_correct = sympy.simplify(expr) == 0.0
                print(sympy.simplify(expr))
            except:
                eq = est._program
                if reverse:
                    eq = f"-({eq})"
                eqC = "Failed"

            print(f"Functional form: {eqC}")

            print(f"Is correct? {is_correct}")

            end = time.time()
            print(f"Evolution finished in {end-start} seconds")

            save_output(filename, trial+1, seed, eq, eqC, is_correct, est._program, weights, loss, target_loss, best_found_loss, target_weights, best_found_weights, end-start)
            
            df = df_append(df, trial+1, seed, eq, eqC, is_correct, est._program, weights, loss, target_loss, best_found_loss, target_weights, best_found_weights, end-start)

            df.to_csv(filename_csv)
