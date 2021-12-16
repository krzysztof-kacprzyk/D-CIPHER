import numpy as np



class EquiPartGrid:

    def __init__(self, widths, samples_per_dim):
        self.num_dims = len(widths)
        self.samples_per_dim = samples_per_dim
        self.widths = widths

        #TODO: change to np.meshgrid
        
        if self.num_dims == 1:
            self.grid = np.mgrid[0:widths[0]:samples_per_dim*1j]
        elif self.num_dims == 2:
            self.grid = np.mgrid[0:widths[0]:samples_per_dim*1j, 0:widths[1]:samples_per_dim*1j]
        elif self.num_dims == 3:
            self.grid = np.mgrid[0:widths[0]:samples_per_dim*1j, 0:widths[1]:samples_per_dim*1j, 0:widths[2]:samples_per_dim*1j]
        elif self.num_dims == 4:
            self.grid = np.mgrid[0:widths[0]:samples_per_dim*1j, 0:widths[1]:samples_per_dim*1j, 0:widths[2]:samples_per_dim*1j, 0:widths[3]:samples_per_dim*1j]
        else:
            raise ValueError("EquiPartGrid supports only dimensions: 1,2,3,4")

        self.shape = [samples_per_dim]*self.num_dims

        
        
    def by_axis(self):
        return self.grid
    
    def as_grid(self):
        return np.moveaxis(self.grid, 0, -1)

    def as_covariates(self):
        return np.moveaxis(self.grid, 0, -1).reshape(self.samples_per_dim ** self.num_dims, self.num_dims)

    def from_labels_to_grid(self, y):
        if len(y) != self.samples_per_dim ** self.num_dims:
            raise ValueError("Wrong dimensions")
        else:
            return y.reshape(self.shape)

    def for_integration(self):
        return np.ones(self.shape) * np.prod(self.widths) / (self.samples_per_dim ** self.num_dims)





# if __name__ == "__main__":
#     g = EquiPartGrid([1,1,1],3)
#     print(g.by_axis())
#     print(g.as_grid())
#     print(g.as_covariates())


