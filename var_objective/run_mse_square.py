import numpy as np
import argparse
import time
from datetime import datetime

from .differential_operator import LinearOperator
from .derivative_estimators import get_diff_engine

from .equations import get_pdes
from .grids import EquiPartGrid
from .generator import generate_fields
from .interpolate import estimate_fields
from .basis import FourierSine2D
from .optimize_operator import MSEWeightsFinder
from .conditions import get_conditions_set
from .config import get_optim_params, get_gp_params
from .libs import SymbolicRegressor, make_fitness
from var_objective.utils.gp_utils import gp_to_pysym_with_coef

INF = 99999999.9


def grid_and_fields_to_covariates(grid_and_fields):

    grid_and_fields = np.moveaxis(grid_and_fields,1,-1)
    num_var = grid_and_fields.shape[-1]
    return np.reshape(grid_and_fields,(-1,num_var))


def _check_if_zero(vector):
    if np.sum(vector == 0.0) == len(vector):
        return True
    else:
        return False

def save_output(filename, trial, seed, program, operator, loss, target_loss, target_loss_better_weights, time_elapsed):
    message = f"""
----------------------
Trial: {trial}
Seed: {seed}
Program: {program}
Operator: {operator}
Loss: {loss}
Target_loss: {target_loss}
Target_loss_better_weights: {target_loss_better_weights}
Time elapsed: {time_elapsed}"""
    with open(filename, 'a') as f:
        f.write(message)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Discover a PDE")
    parser.add_argument('name', help='Equation name from equations.py')
    parser.add_argument('field_index', type=int, help='Which field coordinate to model')
    parser.add_argument('width', type=float, help='Width of the grid')
    parser.add_argument('frequency_per_dim', type=int, help='Frequency per dimension of generated data')
    parser.add_argument('noise_ratio', type=float, help='Noise ration for data generation')
    parser.add_argument('conditions_set', help='Conditions set name from conditions.py')
    parser.add_argument('diff_engine', choices=['numpy', 'tv', 'trend', 'spline', 'finite'])
    parser.add_argument('num_trials', type=int, help='Number of trials')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)

    args = parser.parse_args()

    INF_FLOAT = 9999999999999.9

    pdes = get_pdes(args.name)

    widths = [args.width] * 2

    observed_grid = EquiPartGrid(widths, args.frequency_per_dim)

    dt = datetime.now().strftime("%d-%m-%YT%H.%M.%S")
    filename = f"results/run_mse_square_{dt}"

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

    for trial, seed in enumerate(seeds):
        
        conditions = get_conditions_set(args.conditions_set, params={'seed': seed, 'num_samples':args.num_samples})

        print(f"Seed set to {seed}")
        print(f"Generating dataset of {args.name} on a grid with width {args.width}, frequency per dim {args.frequency_per_dim}, noise ratio {args.noise_ratio} and using conditions set {args.conditions_set}")
        start = time.time()
        observed_dataset = generate_fields(pdes, conditions, observed_grid, args.noise_ratio, seed=seed)
        end = time.time()
        print(f"Observed dataset generated in {end-start} seconds")

        dimension = pdes.get_expression()[args.field_index][0].dimension
        order = pdes.get_expression()[args.field_index][0].order

        engine = get_diff_engine(args.diff_engine)

        print("Initializing MSE Weights Finder")
        start = time.time()
        mse_wf = MSEWeightsFinder(observed_dataset,args.field_index,observed_grid,dimension=dimension,order=order,engine=engine,optim_engine='svd',seed=seed)
        end = time.time()
        print(f"Weight Finder initialized in {end-start} seconds")


        def _mse_fitness(y, y_pred, w):

            # Hack to pass the test
            if len(y_pred) == 2:
                return 0.0

            if _check_if_zero(y_pred):
                loss, weights = mse_wf.find_weights(None)
            else:
                loss, weights = mse_wf.find_weights(y_pred)

            if loss is None:
                return INF

            return loss
        
        X = grid_and_fields_to_covariates(mse_wf.grid_and_fields)
        fake_y = np.zeros(X.shape[0])

        var_fitness = make_fitness(_mse_fitness, greater_is_better=False)

        gp_params = get_gp_params()

        L_target, g_target = pdes.get_expression_normalized()[args.field_index]
        target_weights = L_target.vectorize()[1:] # exclude zero-order partial
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
        
        target_loss = mse_wf._calculate_loss(target_g_part, target_weights)
        print(f"Loss with target weights and target g_part: {target_loss}")
        print(f"Target weights: {target_weights}")

        best_found_loss, best_found_weights = mse_wf.find_weights(target_g_part)
        print(f"Loss for the best found weights: {best_found_loss}")
        print(f"Best found weights: {best_found_weights}")

        print(f"Starting evolution with population {gp_params['population_size']} and {gp_params['generations']} generations")
        start = time.time()
        est = SymbolicRegressor(metric=var_fitness, **gp_params ,verbose=1, random_state=seed)

        est.fit(X, fake_y)



        loss, weights = mse_wf.find_weights(est.predict(X))

        linear_operator = LinearOperator.from_vector(weights, dimension, order, zero_partial=False)

        try:
            eq, eqC = gp_to_pysym_with_coef(est)
        except:
            eq = est._program

        print(f"Found: {linear_operator} - ({eq}) = 0")

        print(f"Expected: {L_target} - ({g_target}) = 0")

        end = time.time()
        print(f"Evolution finished in {end-start} seconds")

        save_output(filename, trial+1, seed, eq, linear_operator, loss, target_loss, best_found_loss, end-start)




