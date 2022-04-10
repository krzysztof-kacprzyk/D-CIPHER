import numpy as np
import matplotlib.pyplot as plt




def get_potential_2D(grid, sources_locs, sources_charges, constant):

    # TODO: check if all sources are outside of the grid

    x = grid.by_axis()

    potentials = []

    for loc, charge in zip(sources_locs,sources_charges):
        distance = np.sqrt((x[0]-loc[0]) ** 2 + (x[1]-loc[1]) ** 2)
        potentials.append(constant * charge / distance)
    
    potential = np.sum(np.stack(potentials,axis=0),axis=0)
    return potential

def get_potential_3D(grid, sources_locs, sources_charges, constant):

    # TODO: check if all sources are outside of the grid

    x = grid.by_axis()

    potentials = []

    for loc, charge in zip(sources_locs,sources_charges):
        distance = np.sqrt((x[0]-loc[0]) ** 2 + (x[1]-loc[1]) ** 2 + (x[2]-loc[2]) ** 2)
        potentials.append(constant * charge / distance)
    
    potential = np.sum(np.stack(potentials,axis=0),axis=0)
    return potential






