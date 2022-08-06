import numpy as np

U = 0.3
V = 0.3

def get_flow_potential_2D(grid, sources_locs, sources_strengths):
     # TODO: check if all sources are outside of the grid

    x = grid.by_axis()

    potentials = []

    for loc, strength in zip(sources_locs,sources_strengths):
        distance = np.sqrt((x[0]-loc[0]) ** 2 + (x[1]-loc[1]) ** 2)

        potentials.append(strength * np.log(distance) / (2*np.pi))
    
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    theta = np.arctan2(x[1], x[0])
    potential = np.sum(np.stack(potentials,axis=0),axis=0) + U*(r**3)*np.cos(3*theta) + V*(r**2)*np.cos(2*theta)
    return potential

def get_flow_stream_2D(grid, sources_locs, sources_strengths):
     # TODO: check if all sources are outside of the grid

    x = grid.by_axis()

    stream_fs = []

    for loc, strength in zip(sources_locs,sources_strengths):
        distance = np.sqrt((x[0]-loc[0]) ** 2 + (x[1]-loc[1]) ** 2)
        x_disp = x[0] - loc[0]
        y_disp = x[1] - loc[1]
        angle  = np.arctan2(y_disp, x_disp)
        # angle = np.where(angle < 0, 2*np.pi + angle, angle)
        
        stream_fs.append(strength * angle / (2*np.pi))

    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    theta = np.arctan2(x[1], x[0])
    stream_f = np.sum(np.stack(stream_fs,axis=0),axis=0) + U*(r**3)*np.sin(3*theta) + V*(r**2)*np.sin(2*theta)
    return stream_f


def get_complex_potential_re(grid, source_locs, sources_strengths):
    x = grid.by_axis()

    cx = x[0] + x[1]*1j
    s1 = source_locs[0]
    s2 = source_locs[1]
    w = sources_strengths[0] * cx ** 2 + sources_strengths[1] * np.log(cx - s1[0] - s1[1]*1j) + (cx - s2[0] - s2[1]*1j)**(-1)

    return np.real(w)

def get_complex_potential_im(grid, source_locs, sources_strengths):
    x = grid.by_axis()

    cx = x[0] + x[1]*1j
    s1 = source_locs[0]
    s2 = source_locs[1]
    w = sources_strengths[0] * cx ** 2 + sources_strengths[1] * np.log(cx - s1[0] - s1[1]*1j) + (cx - s2[0] - s2[1]*1j)**(-1)

    return np.imag(w)

