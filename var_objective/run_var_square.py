from datetime import datetime
import numpy as np
import argparse
import time
import pandas as pd
import pickle
import os

from var_objective.differential_operator import LinearOperator
from var_objective.utils.gp_utils import gp_to_pysym_no_coef, gp_to_pysym_with_coef
from var_objective.utils.logging import create_output_dir, create_logging_file_names

from .equations import get_pdes
from .grids import EquiPartGrid
from .generator import generate_fields
from .interpolate import estimate_fields
from .basis import BSplineFreq2D, BSplineTrans1D, FourierSine2D, BSplineTrans2D
from .optimize_operator import VariationalWeightsFinder
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

def save_output(filename, trial, seed, program, eqC, is_correct, raw_program, operator, loss, target_loss, target_loss_better_weights, target_weights, best_found_weights, time_elapsed, time_preprocessing):
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
Time elapsed: {time_elapsed}
Time preprocessing: {time_preprocessing}"""
    with open(filename, 'a') as f:
        f.write(message)

def df_append(old_df, trial, seed, program, eqC, is_correct, raw_program, operator, loss, target_loss, target_loss_better_weights, target_weights, best_found_weights, time_elapsed, time_preprocessing):
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
    df['time_preprocessing'] = [time_preprocessing]
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
    parser.add_argument('field_index', type=int, help='Which field coordinate to model')
    parser.add_argument('width', type=float, help='Width of the grid')
    parser.add_argument('frequency_per_dim', type=int, help='Frequency per dimension of generated data')
    parser.add_argument('noise_ratio', type=float, help='Noise ration for data generation')
    parser.add_argument('full_grid_samples', type=int, help='Frequency of the full grid')
    parser.add_argument('conditions_set', help='Conditions set name from conditions.py')
    parser.add_argument('basis', choices=['fourier','2spline2D', '2spline2Dtrans', '2spline1Dtrans'])
    parser.add_argument('max_ind_basis', type=int, help='Maximum index for test functions. Number of used test functions is a square of this number')
    parser.add_argument('num_trials', type=int, help='Number of trials')
    parser.add_argument('normalization',choices=['l1','l2'])
    parser.add_argument('solver', help='Least squares solver')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--warm_start', type=int, default=1)
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

    output_dir = create_output_dir(args.exp_name, 'var')

    dt = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    filename, filename_csv, filename_meta = create_logging_file_names(output_dir, dt)

    gp_params = get_gp_params(generations=args.generations)

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

        start_preprocessing = time.time()

        print(f'Trial {trial+1}/{len(seeds)}')

        if trial+1 < args.warm_start:
            print("skipping")
            continue

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

        dimension = pdes.get_expression()[args.field_index][0].dimension
        order = pdes.get_expression()[args.field_index][0].order

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

        print("Initializing Variational Weights Finder")
        start = time.time()
        var_wf = VariationalWeightsFinder(full_dataset,args.field_index,full_grid,dimension=dimension,order=order,basis=basis,index_limits=index_limits,optim_engine=LSTSQ_SOLVER,seed=seed)
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
       

        L_target, g_target = pdes.get_expression_normalized(norm=args.normalization)[args.field_index]
        target_weights = L_target.get_adjoint().vectorize()[1:] # exclude zero-order partial
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
        print(f"Loss with target weights and target g_part: {target_loss}")
        print(f"Target weights: {target_weights}")

        best_found_loss, best_found_weights = var_wf.find_weights(target_g_part)
        print(f"Loss for the best found weights: {best_found_loss}")
        print(f"Best found weights: {best_found_weights}")

        end_preprocessing = time.time()

        print(f"Starting evolution with population {gp_params['population_size']} and {gp_params['generations']} generations")
        start = time.time()
        est = SymbolicRegressor(metric=var_fitness, **gp_params ,verbose=1, random_state=seed)

        est.fit(X, fake_y)

        loss, weights = var_wf.find_weights(est.predict(X))
        linear_operator = LinearOperator.from_vector(weights, dimension, order, zero_partial=False).get_adjoint()
        reverse = False
        if linear_operator.get_sign() < 0:
            reverse = True
            linear_operator = linear_operator.reverse_sign()
        is_correct = "NA"
        try:
            eq, eqC = gp_to_pysym_with_coef(est)
            if reverse:
                eq = (-1) * eq
                eqC = (-1) * eqC
            true_g = pdes.get_functional_form_normalized(norm=args.normalization)[args.field_index]
            expr = sympy.parsing.sympy_parser.parse_expr(f"{true_g} - ({eqC})")
            is_correct = sympy.simplify(expr) == 0.0
            print(sympy.simplify(expr))
        except:
            eq = est._program
            if reverse:
                eq = f"-({eq})"
            eqC = "Failed"

        print(f"Found: {linear_operator} - ({eq}) = 0")

        print(f"Expected: {L_target} - ({g_target}) = 0")

        print(f"Functional form: {eqC}")

        print(f"Is correct? {is_correct}")

        end = time.time()
        print(f"Evolution finished in {end-start} seconds")

        save_output(filename, trial+1, seed, eq, eqC, is_correct, est._program, linear_operator, loss, target_loss, best_found_loss, target_weights, best_found_weights, end-start, end_preprocessing-start_preprocessing)
        
        df = df_append(df, trial+1, seed, eq, eqC, is_correct, est._program, linear_operator, loss, target_loss, best_found_loss, target_weights, best_found_weights, end-start, end_preprocessing-start_preprocessing)

        df.to_csv(filename_csv)
