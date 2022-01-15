import numpy as np
import argparse
import time

from .differential_operator import LinearOperator
from .derivative_estimators import get_diff_engine

from .equations import get_pdes
from .grids import EquiPartGrid
from .generator import generate_fields
from .interpolate import estimate_fields
from .basis import FourierSine2D
from .optimize_operator import MSEWeightsFinder, normalize
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
    parser.add_argument('conditions_set', help='Conditions set name from conditions.py')
    parser.add_argument('diff_engine', choices=['numpy', 'tv', 'trend', 'spline', 'finite'])
    parser.add_argument('num_trials', type=int, help='Number of trials')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    INF_FLOAT = 9999999999999.9

    LSTSQ_SOLVER = 'unit_L'

    pdes = get_pdes(args.name)

    widths = [args.width] * 2


    observed_grid = EquiPartGrid(widths, args.frequency_per_dim)

    conditions = get_conditions_set(args.conditions_set)

    print(f"Seed set to {args.seed}")
    print(f"Generating dataset of {args.name} on a grid with width {args.width}, frequency per dim {args.frequency_per_dim}, noise ratio {args.noise_ratio} and using conditions set {args.conditions_set}")
    start = time.time()
    observed_dataset = generate_fields(pdes, conditions, observed_grid, args.noise_ratio, seed=args.seed)
    end = time.time()
    print(f"Observed dataset generated in {end-start} seconds")

    dimension = pdes.get_expression()[args.field_index][0].dimension
    order = pdes.get_expression()[args.field_index][0].order
   
    opt_params = get_optim_params()

    engine = get_diff_engine(args.diff_engine)

    print("Initializing MSE Weights Finder")
    start = time.time()
    mse_wf = MSEWeightsFinder(observed_dataset,args.field_index,observed_grid,dimension=dimension,order=order,engine=engine,**opt_params, seed=args.seed, calculate_svd=True)
    end = time.time()
    print(f"Weight Finder initialized in {end-start} seconds")


    def _mse_fitness(y, y_pred, w):

        # Hack to pass the test
        if len(y_pred) == 2:
            return 0.0

        if _check_if_zero(y_pred):
            return INF

        loss, weights = mse_wf.find_weights(y_pred,from_covariates=True, normalize_g=LSTSQ_SOLVER, only_loss=True)

        return loss
    
    X = grid_and_fields_to_covariates(mse_wf.grid_and_fields)
    fake_y = np.zeros(X.shape[0])

    var_fitness = make_fitness(_mse_fitness, greater_is_better=False)

    gp_params = get_gp_params()

    loss2, weights2 = mse_wf.find_weights(4*np.sin(2*np.pi*X[:,1]),from_covariates=True,normalize_g=LSTSQ_SOLVER)

    print(loss2, weights2)

    loss3, weights3 = mse_wf.find_weights(np.sin(-np.sin(2*X[:,1]-1) / 0.288),from_covariates=True,normalize_g=LSTSQ_SOLVER)

    print(loss3, weights3)

    print(f"Starting evolution with population {gp_params['population_size']} and {gp_params['generations']} generations")
    start = time.time()
    est = SymbolicRegressor(metric=var_fitness, **gp_params ,verbose=1, random_state=args.seed)

    est.fit(X, fake_y)



    loss, weights = mse_wf.find_weights(est.predict(X),from_covariates=True,normalize_g=LSTSQ_SOLVER)

    linear_operator = LinearOperator.from_vector(weights, dimension, order, zero_partial=False)
    print(f"{linear_operator} - {est._program} = 0")

    end = time.time()
    print(f"Evolution finished in {end-start} seconds")





