from .equations import get_pdes
from .grids import EquiPartGrid
from .generator import generate_fields
from .interpolate import estimate_fields
import numpy as np
import matplotlib.pyplot as plt

NAME = "TestEquation1"
SEED = 0

pdes = get_pdes(NAME)

widths = [1.0, 1.0]

num_samples = 1
frequency_per_dim = 5
noise_ratio = 0.3
integration_samples = 1000

observed_grid = EquiPartGrid(widths, frequency_per_dim)
full_grid = EquiPartGrid(widths, integration_samples)

observed_dataset = generate_fields(pdes.get_solution([lambda t: np.sin(t*np.pi)]), observed_grid, num_samples, noise_ratio, seed=SEED)

print(observed_dataset.shape)
print(observed_dataset)

estimated_dataset = estimate_fields(observed_grid, observed_dataset, full_grid, seed=0)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(full_grid.by_axis()[0], full_grid.by_axis()[1], estimated_dataset[0][0],
                       linewidth=0, antialiased=False)

ax.scatter(observed_grid.by_axis()[0], observed_grid.by_axis()[1], observed_dataset[0][0], color='red')

plt.show()