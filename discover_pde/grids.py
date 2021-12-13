import numpy as np



class UniformGrid:

    def generate(dimensions, samples_per_dim):
        n = len(dimensions)
        
        if n == 1:
            return np.mgrid[0:dimensions[0]:samples_per_dim*1j]
        elif n == 2:
            return np.mgrid[0:dimensions[0]:samples_per_dim*1j, 0:dimensions[1]:samples_per_dim*1j]
        elif n == 3:
            return np.mgrid[0:dimensions[0]:samples_per_dim*1j, 0:dimensions[1]:samples_per_dim*1j, 0:dimensions[2]:samples_per_dim*1j]
        elif n == 4:
            return np.mgrid[0:dimensions[0]:samples_per_dim*1j, 0:dimensions[1]:samples_per_dim*1j, 0:dimensions[2]:samples_per_dim*1j, 0:dimensions[3]:samples_per_dim*1j]
        else:
            raise ValueError("UniformGrid supports only dimensions: 1,2,3,4")
        
