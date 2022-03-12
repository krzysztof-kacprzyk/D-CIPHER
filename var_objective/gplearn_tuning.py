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
from .optimize_operator import MSEWeightsFinder
from .conditions import get_conditions_set
from .config import get_optim_params, get_gp_params
from .libs import SymbolicRegressor, make_fitness
from .utils.gp_utils import gp_to_pysym_with_coef

from sklearn.model_selection import ParameterGrid
import pickle
from datetime import datetime
import sympy
import pandas as pd
from copy import deepcopy

INF_FLOAT = 0.9e+300

def grid_and_fields_to_covariates(grid_and_fields):

    grid_and_fields = np.moveaxis(grid_and_fields,1,-1)
    num_var = grid_and_fields.shape[-1]
    return np.reshape(grid_and_fields,(-1,num_var))


def _check_if_zero(vector):
    if np.sum(vector == 0.0) == len(vector):
        return True
    else:
        return False

def log_print(msg):
    print(msg)
    log_file_name = "log.txt" 
    with open(log_file_name, "a") as f:
        f.write(f"{msg}\n")


def save_checkpoint(_dt, seed, _results, _used_params, _programmes, _programme_lengths):
    min_i = np.argmin(_results)
    log_print(f"Smallest loss: {_results[min_i]} with parameters {_used_params[min_i]}")
    to_save = (_results, _used_params, _programmes, _programme_lengths)
    name_pickle = f"results/gplearn_tuning_{_dt}.p"
    name_txt = f"results/gplearn_tuning_{_dt}.txt"
    name_csv = f"results/gplearn_tuning_{_dt}.csv"

    pickle.dump(to_save, open(name_pickle, "wb" ))
    with open(name_txt, "w") as f:
        f.write(f"""Seed: {seed}\n
        Num of programmes: {len(_results)}\n
        Loss: {_results[min_i]}\n
        Parameters: {_used_params[min_i]}\n
        Function: {_programmes[min_i]}\n
        Length: {_programme_lengths[min_i]}\n
        PDE name: {PDES_NAME}\n
        WIDTH: {WIDTH}\n
        FREQUENCY_PER_DIM: {FREQUENCY_PER_DIM}\n
        NOISE_RATIO: {NOISE_RATIO}\n
        CONDITIONS_SET: {CONDITIONS_SET}\n
        DIFF_ENGINE: {DIFF_ENGINE}\n
        NUM_TESTS: {NUM_TESTS}\n
        ARGS: {vars(args)}""")
    df = pd.DataFrame(_used_params)
    df['loss'] = np.array(_results)
    df['equation'] = _programmes
    df.to_csv(name_csv)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Discover a PDE")
    parser.add_argument('width', type=float, help='Width of the grid')
    parser.add_argument('frequency_per_dim', type=int, help='Frequency per dimension of generated data')
    parser.add_argument('num_tests', type=int, help='Number of trials')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--population_size', nargs='+', default=[1000], type=int)
    parser.add_argument('--generations', nargs='+', default=[20], type=int)
    parser.add_argument('--tournament_size', nargs='+', default=[20], type=int)
    parser.add_argument('--parsimony_coefficient', nargs='+', default=[0.001], type=float)
    # parser.add_argument('--keep_probs_fixed',action='store_true')

    args = parser.parse_args()

    PDES_NAME = 'HeatEquation_0.1'
    WIDTH = args.width
    FREQUENCY_PER_DIM = args.frequency_per_dim
    NOISE_RATIO = 0
    CONDITIONS_SET = 'HeatTuning'
    FIELD_INDEX = 0
    DIFF_ENGINE = 'finite'
    NUM_TESTS = args.num_tests

    pdes = get_pdes(PDES_NAME)

    widths = [WIDTH] * 2

    SEED = args.seed


    observed_grid = EquiPartGrid(widths, FREQUENCY_PER_DIM)

    conditions = get_conditions_set(CONDITIONS_SET)

    log_print(f"Seed set to {SEED}")
    log_print(f"Generating dataset of {PDES_NAME} on a grid with width {WIDTH}, frequency per dim {FREQUENCY_PER_DIM}, noise ratio {NOISE_RATIO} and using conditions set {CONDITIONS_SET}")
    start = time.time()
    observed_dataset = generate_fields(pdes, conditions, observed_grid, NOISE_RATIO, seed=SEED)
    end = time.time()
    log_print(f"Observed dataset generated in {end-start} seconds")

    dimension = pdes.get_expression()[FIELD_INDEX][0].dimension
    order = pdes.get_expression()[FIELD_INDEX][0].order
   
    opt_params = get_optim_params()

    engine = get_diff_engine(DIFF_ENGINE)

    log_print("Initializing MSE Weights Finder")
    start = time.time()
    mse_wf = MSEWeightsFinder(observed_dataset,FIELD_INDEX,observed_grid,dimension=dimension,order=order,engine=engine,**opt_params, seed=SEED)
    end = time.time()
    log_print(f"Weight Finder initialized in {end-start} seconds")


    def _mse_fitness(y, y_pred, w):

        # Hack to pass the test
        if len(y_pred) == 2:
            return 0.0

        if _check_if_zero(y_pred):
            loss, weights = mse_wf.find_weights(None,only_loss=True)
        else:
            loss, weights = mse_wf.find_weights(y_pred,only_loss=True)

        if loss is None:
            return INF_FLOAT

        return loss
    
    X = grid_and_fields_to_covariates(mse_wf.grid_and_fields)
    fake_y = np.zeros(X.shape[0])

    var_fitness = make_fitness(_mse_fitness, greater_is_better=False)

    gp_parameters_values = {
        'population_size':args.population_size,
        'generations':args.generations,
        'tournament_size':args.tournament_size,
        'p_crossover':[0.6,0.7,0.8,0.9],
        'p_subtree_mutation':[0.01,0.05,0.1,0.15],
        'p_hoist_mutation':[0.01,0.02,0.05,0.07],
        'p_point_mutation':[0.01,0.05,0.1,0.15],
        'parsimony_coefficient':args.parsimony_coefficient,
        'patience':[20]
    }

    np.random.seed(SEED)

    # seeds = [np.random.randint(0,999999999) for i in range(NUM_TESTS)]

    parameter_grid = ParameterGrid(gp_parameters_values)

    params_list = np.random.choice(list(parameter_grid), size=NUM_TESTS, replace=False)

    results = []
    used_params = []
    programmes = []
    weights = []
    programme_lengths = []

    dt = datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    for index, param in enumerate(params_list):


        log_print(f"Experiment {index+1} out of {len(params_list)}")
        log_print(f"Using {param}")

        # param = deepcopy(_param)

        if param['p_crossover'] + param['p_subtree_mutation'] + param['p_hoist_mutation'] + param['p_point_mutation'] >= 1.0:
            log_print("This parameters are invalid. No experiment will be carried out")
            continue
        
        start = time.time()

        # np.random.seed(seeds[index])

        # if args.keep_probs_fixed:
        #     p_cross = 0.9
        #     p_subtree = 0.01
        #     p_hoist = 0.01
        #     p_point = 0.01
        # else:
        #     sum = 2
        #     while sum > 1:
        #         p_cross  = np.random.rand()
        #         p_subtree = np.random.rand()
        #         p_hoist = np.random.rand()
        #         p_point = np.random.rand() 
        #         sum = p_cross + p_subtree + p_hoist + p_point
            
        # param['p_crossover'] = p_cross
        # param['p_subtree_mutation'] = p_subtree
        # param['p_hoist_mutation'] = p_hoist
        # param['p_point_mutation'] = p_point

        est = SymbolicRegressor(metric=var_fitness, **param ,verbose=1, random_state=SEED, function_set=('add', 'sub', 'mul', 'div','sin', 'log','exp'), n_jobs=-1, best_so_far=True)
        est.fit(X, fake_y)
        
        loss, weights = mse_wf.find_weights(est.predict(X),only_loss=False)
        log_print(est._program)
        try:
            eq, eqC = gp_to_pysym_with_coef(est)
        except:
            eq = est._program
        results.append(loss)
        used_params.append(param)
        programmes.append(f"{eq}")
        programme_lengths.append(est._program.length_)

        end = time.time()

        log_print(f"{weights} - ({eq}) = 0")
        log_print(f"The evolution took {end-start} seconds")

        save_checkpoint(dt, SEED, results, used_params, programmes, programme_lengths)

