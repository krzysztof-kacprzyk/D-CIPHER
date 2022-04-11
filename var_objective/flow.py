import numpy as np


def get_flow_potential_2D(grid, sources_locs, sources_strengths):
     # TODO: check if all sources are outside of the grid

    x = grid.by_axis()

    potentials = []

    for loc, strength in zip(sources_locs,sources_strengths):
        distance = np.sqrt((x[0]-loc[0]) ** 2 + (x[1]-loc[1]) ** 2)
        potentials.append(strength * np.log(distance) / (2*np.pi))
    
    potential = np.sum(np.stack(potentials,axis=0),axis=0)
    return potential