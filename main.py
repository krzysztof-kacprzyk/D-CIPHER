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
from var_objective.optimize_operator import MSEWeightsFinder
from var_objective.differential_operator import all_derivatives, NumpyDiff
import numpy as np
import matplotlib.pyplot as plt

NAME = "TestEquation1"
SEED = 0

pdes = get_pdes(NAME)

widths = [1.0, 1.0]

num_samples = 1
frequency_per_dim = 100
noise_ratio = 0.1
full_grid_samples = 1000

observed_grid = EquiPartGrid(widths, frequency_per_dim)
#full_grid = EquiPartGrid(widths, full_grid_samples)

conditions = Conditions(1)
conditions.add_sample([lambda t: np.sin(t*np.pi)])

observed_dataset = generate_fields(pdes, conditions, observed_grid, noise_ratio, seed=SEED)

mse_wf = MSEWeightsFinder(observed_dataset,0,observed_grid,2,1,NumpyDiff(),alpha=1.0,beta=0.0,optim_params={'lr':0.01},num_epochs=200)

# mse_wf.find_weights(g_part=None)

# derivative_fields = all_derivatives(observed_dataset[0][0],observed_grid,2,1,NumpyDiff())
# print(derivative_fields)

#TODO: incorporate w somehow

def grid_and_fields_to_covariates(grid_and_fields):

    grid_and_fields = np.moveaxis(grid_and_fields,1,-1)
    num_var = grid_and_fields.shape[-1]
    return np.reshape(grid_and_fields,(-1,num_var))





def _mse_fitness(y, y_pred, w):
    if len(y_pred) == 2:
        print("Test")
        return 0.0
    g_part = np.reshape(y_pred, ((mse_wf.D, *(mse_wf.grid.shape))))

    loss, weights = mse_wf.find_weights(g_part)
    return loss.item()

X = grid_and_fields_to_covariates(mse_wf.grid_and_fields)
fake_y = np.zeros(X.shape[0])

print(X.shape)

mse_fitness = make_fitness(_mse_fitness, greater_is_better=False)

est = SymbolicRegressor(metric=mse_fitness, population_size=10)



est.fit(X, fake_y)
print(est._program)

_mse_fitness(np.zeros(10), est.predict(X), np.zeros(10))