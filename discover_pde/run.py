from torch import full
from .equations import get_pdes
from .grids import EquiPartGrid
from .generator import generate_fields
from .interpolate import estimate_fields
from .basis import FourierSine2D
from .optimize_operator import LinearOperatorFinder
import numpy as np
import matplotlib.pyplot as plt

NAME = "TestEquation1"
SEED = 0

pdes = get_pdes(NAME)

widths = [1.0, 1.0]

num_samples = 1
frequency_per_dim = 5
noise_ratio = 0.3
full_grid_samples = 1000

observed_grid = EquiPartGrid(widths, frequency_per_dim)
full_grid = EquiPartGrid(widths, full_grid_samples)

observed_dataset = generate_fields(pdes.get_solution([lambda t: np.sin(t*np.pi)]), observed_grid, num_samples, noise_ratio, seed=SEED)

print(observed_dataset.shape)
print(observed_dataset)

estimated_dataset = estimate_fields(observed_grid, observed_dataset, full_grid, seed=SEED)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(full_grid.by_axis()[0], full_grid.by_axis()[1], estimated_dataset[0][0],
#                        linewidth=0, antialiased=False)

# ax.scatter(observed_grid.by_axis()[0], observed_grid.by_axis()[1], observed_dataset[0][0], color='red')

# plt.show()
optim_params = {
    'lr': 0.05
}
lof = LinearOperatorFinder(estimated_dataset, 0, full_grid, 2, 1, FourierSine2D(widths), [6,6], alpha=0.5, beta=0.3, optim_name='sgd', optim_params=optim_params,num_epochs=200, patience=20,seed=SEED)
weights = lof.find_weights(None)
