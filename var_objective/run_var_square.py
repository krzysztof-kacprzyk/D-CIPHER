import numpy as np
import argparse
import time

from var_objective.differential_operator import LinearOperator
from var_objective.utils.gp_utils import gp_to_pysym_with_coef

from .equations import get_pdes
from .grids import EquiPartGrid
from .generator import generate_fields
from .interpolate import estimate_fields
from .basis import BSplineFreq2D, FourierSine2D
from .optimize_operator import VariationalWeightsFinder
from .conditions import get_conditions_set
from .config import get_optim_params, get_gp_params
from .libs import SymbolicRegressor, make_fitness

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Discover a PDE")
    parser.add_argument('name', help='Equation name from equations.py')
    parser.add_argument('field_index', type=int, help='Which field coordinate to model')
    parser.add_argument('width', type=float, help='Width of the grid')
    parser.add_argument('frequency_per_dim', type=int, help='Frequency per dimension of generated data')
    parser.add_argument('noise_ratio', type=float, help='Noise ration for data generation')
    parser.add_argument('full_grid_samples', type=int, help='Frequency of the full grid')
    parser.add_argument('conditions_set', help='Conditions set name from conditions.py')
    parser.add_argument('basis', choices=['fourier','2spline2D'])
    parser.add_argument('max_ind_basis', type=int, help='Maximum index for test functions. Number of used test functions is a square of this number')
    parser.add_argument('num_trials', type=int, help='Number of trials')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    INF_FLOAT = 9999999999999.9

    pdes = get_pdes(args.name)

    widths = [args.width] * 2

    observed_grid = EquiPartGrid(widths, args.frequency_per_dim)
    full_grid = EquiPartGrid(widths, args.full_grid_samples)

    conditions = get_conditions_set(args.conditions_set)

    print(f"Seed set to {args.seed}")
    print(f"Generating dataset of {args.name} on a grid with width {args.width}, frequency per dim {args.frequency_per_dim}, noise ratio {args.noise_ratio} and using conditions set {args.conditions_set}")
    start = time.time()
    observed_dataset = generate_fields(pdes, conditions, observed_grid, args.noise_ratio, seed=args.seed)
    end = time.time()
    print(f"Observed dataset generated in {end-start} seconds")

    print(f"Estimating fields on a grid with frequency {args.full_grid_samples} per dimension")
    start = time.time()
    full_dataset = estimate_fields(observed_grid,observed_dataset,full_grid,seed=args.seed)
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
   
    opt_params = get_optim_params()

    print("Initializing Variational Weights Finder")
    start = time.time()
    var_wf = VariationalWeightsFinder(full_dataset,args.field_index,full_grid,dimension=dimension,order=order,basis=basis,index_limits=index_limits,**opt_params, seed=args.seed)
    end = time.time()
    print(f"Weight Finder initialized in {end-start} seconds")


    def _var_fitness(y, y_pred, w):

        if len(y_pred) == 2:
            return 0.0

        if _check_if_zero(y_pred):
            loss, weights = var_wf.find_weights(None, only_loss=True)
        else:
            loss, weights = var_wf.find_weights(y_pred, only_loss=True)

        if loss is None:
            return INF

        return loss
    
    X = grid_and_fields_to_covariates(var_wf.grid_and_fields)
    fake_y = np.zeros(X.shape[0])

    var_fitness = make_fitness(_var_fitness, greater_is_better=False)

    gp_params = get_gp_params()

    L_target, g_target = pdes.get_expression_normalized()[args.field_index]
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

    best_found_loss, best_found_weights = var_wf.find_weights(target_g_part, only_loss=False)
    print(f"Loss for the best found weights: {best_found_loss}")
    print(f"Best found weights: {best_found_weights}")

    print(f"Starting evolution with population {gp_params['population_size']} and {gp_params['generations']} generations")
    start = time.time()
    est = SymbolicRegressor(metric=var_fitness, **gp_params ,verbose=1, random_state=args.seed)

    est.fit(X, fake_y)

    loss, weights = var_wf.find_weights(est.predict(X), only_loss=False)

    linear_operator = LinearOperator.from_vector(weights, dimension, order, zero_partial=False)

  
    try:
        eq, eqC = gp_to_pysym_with_coef(est)
    except:
        eq = est._program

    print(f"Found {linear_operator.get_adjoint()} - {eq} = 0")

    print(f"Expected: {L_target} - {g_target} = 0")

    end = time.time()
    print(f"Evolution finished in {end-start} seconds")
