from tvregdiff.tvregdiff import TVRegDiff

from gplearn.gplearn.fitness import make_fitness
from gplearn.gplearn.genetic import SymbolicRegressor

import numpy as np
import matplotlib.pyplot as plt

# t = np.linspace(0,1,100)
# x = t ** 2

# dt = t[1] - t[0]

# print(x)

# dxdt = TVRegDiff(x, 50, 0.1, dx=dt, precondflag=False, plotflag=False)

# print(dxdt)

# data = np.loadtxt('tvregdiff/test_data.dat')
# X = np.linspace(0,8,50) 
# Y = np.sin(X) + np.random.normal(0,0.1,50)

# dX = X[1] - X[0]

# dYdX = TVRegDiff(Y, 20, 5e-2, dx=dX, ep=1e-5,
#                     scale='small',
#                     plotflag=False, precondflag=True,
#                     diffkernel='sq')

# plt.plot(X, Y, label='f(x)', c=(0.2, 0.2, 0.2), lw=0.5)
# plt.plot(X, np.gradient(Y, dX),
#             label='df/dx (numpy)', c=(0, 0.3, 0.8), lw=1)
# plt.plot(X, dYdX, label='df/dx (TVRegDiff)',
#             c=(0.8, 0.3, 0.0), lw=1)
# plt.legend()
# plt.show()

from var_objective.equations import get_pdes
from var_objective.grids import EquiPartGrid
from var_objective.generator import generate_fields, Conditions
from var_objective.interpolate import estimate_fields
from var_objective.basis import FourierSine2D
from var_objective.optimize_operator import MSEWeightsFinder, VariationalWeightsFinder
from var_objective.differential_operator import all_derivatives, NumpyDiff
import numpy as np
import matplotlib.pyplot as plt

NAME = "TestEquation2"
SEED = 0
INF_FLOAT = 9999999999999.9
pdes = get_pdes(NAME)

widths = [3.0, 3.0]

frequency_per_dim = 20
noise_ratio = 0.2
full_grid_samples = 500

observed_grid = EquiPartGrid(widths, frequency_per_dim)
full_grid = EquiPartGrid(widths, full_grid_samples)

conditions = Conditions(1)
conditions.add_sample([lambda t: np.sin(t)])
conditions.add_sample([lambda t: np.power(t,2.0)])
conditions.add_sample([lambda t: t])


observed_dataset = generate_fields(pdes, conditions, observed_grid, noise_ratio, seed=SEED)
print("Observed dataset generated")

full_dataset = estimate_fields(observed_grid,observed_dataset,full_grid,seed=SEED)
print("Fields estimated")


# mse_wf = MSEWeightsFinder(observed_dataset,0,observed_grid,2,1,NumpyDiff(),alpha=1.0,beta=0.2,optim_name='sgd',optim_params={'lr':0.01},num_epochs=300,patience=20)
var_wf = VariationalWeightsFinder(full_dataset,0,full_grid,dimension=2,order=1,basis=FourierSine2D(widths),index_limits=[6,6],alpha=0.0,beta=0.0,optim_name='adam',optim_params={'lr':0.5},num_epochs=300,patience=30)
print("Weight Finder initialized")




def grid_and_fields_to_covariates(grid_and_fields):

    grid_and_fields = np.moveaxis(grid_and_fields,1,-1)
    num_var = grid_and_fields.shape[-1]
    return np.reshape(grid_and_fields,(-1,num_var))


def _check_if_zero(vector, tol):
    if np.sum(vector < tol) == len(vector):
        return True
    else:
        return False




#TODO: incorporate w somehow
def _mse_fitness(y, y_pred, w):
    if len(y_pred) == 2:
        print("Test")
        return 0.0

    loss, weights = mse_wf.find_weights(y_pred,from_covariates=True)

    return loss

def _var_fitness(y, y_pred, w):
    if len(y_pred) == 2:
        print("Test")
        return 0.0

    loss, weights = var_wf.find_weights(y_pred,from_covariates=True)

    return loss

X = grid_and_fields_to_covariates(var_wf.grid_and_fields)
fake_y = np.zeros(X.shape[0])

print(X.shape)

# mse_fitness = make_fitness(_mse_fitness, greater_is_better=False)
var_fitness = make_fitness(_var_fitness, greater_is_better=False)

est = SymbolicRegressor(metric=var_fitness, population_size=10,  function_set=('add', 'sub', 'mul', 'div','sin'),verbose=1)



est.fit(X, fake_y)
print(est._program)


loss, weights = var_wf.find_weights(est.predict(X),from_covariates=True,normalize_g=False)
print(f"Loss: {loss}, Weights: {weights}")

target_loss, target_weights = var_wf.find_weights(np.sin(X[:,1]),from_covariates=True,normalize_g=False)
print(f"Target Loss: {target_loss}, Target Weights: {target_weights}")
