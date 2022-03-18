import numpy as np
from itertools import product

from var_objective.grids import EquiPartGrid
from .differential_operator import LinearOperator
from .derivative_estimators import all_derivatives
from .utils.lstsq_solver import UnitLstsqSDR, UnitLstsqSVD

class VariationalWeightsFinder:

    def __init__(self,estimated_dataset, field_index, full_grid, dimension, order, basis, index_limits, optim_engine='svd', seed=0):
        self.estimated_dataset = estimated_dataset
        self.field_index = field_index
        self.full_grid = full_grid
        self.dimension = dimension
        self.order = order
        self.basis = basis
        self.index_limits = index_limits
        self.seed = seed


        np.random.seed(self.seed)

        if len(self.index_limits) != self.basis.num_indexes:
            raise ValueError("Length of index_limits does not match num_indexes of basis")

        #TODO: check dataset shape and full_grid shape

        basis_indexes = list(product(*[range(1, self.index_limits[i]+1) for i in range(len(self.index_limits))]))
        self.S = len(basis_indexes)

        all_partials = LinearOperator.get_all_partials(self.dimension, self.order)
        self.J = len(all_partials)

        self.test_function_part = np.zeros((self.S,self.J,*self.full_grid.shape))
        for j, partial in enumerate(all_partials):
            for s, indexes in enumerate(basis_indexes):
                self.test_function_part[s][j] = self.basis.get_tensor(indexes, self.full_grid, partial)

        self.D = self.estimated_dataset.shape[0]
        self.N = self.estimated_dataset.shape[1]
        self.M = self.full_grid.num_dims
        self.J = LinearOperator.get_vector_length(self.dimension, self.order)

        if self.field_index >= self.N:
            raise ValueError(f"There is no field with index {self.field_index}")
            
        static_part = np.zeros([self.D,self.S,self.J,*self.full_grid.shape])

        for d in range(self.D):
            static_part[d] = np.multiply(self.test_function_part, self.estimated_dataset[d][field_index])
        
        if isinstance(self.full_grid, EquiPartGrid):
            integrals = static_part.sum(axis=tuple(range(3, len(static_part.shape)))) * self.full_grid.get_integration_constant()
        else:
            integrals = np.multiply(static_part, self.full_grid.for_integration()).sum(axis=tuple(range(3, len(static_part.shape))))

        assert integrals.shape == (self.D, self.S, self.J)

        assert self.order > 0
        self.num_lower_order = LinearOperator.get_vector_length(self.dimension, self.order-1)
        

        self.grid_and_fields = np.zeros([self.D,(self.M + self.N), *full_grid.shape])
        for d in range(self.D):
            self.grid_and_fields[d] = np.stack([*full_grid.by_axis(),*(estimated_dataset[d])],axis=0)

        self.X = np.reshape(integrals,(-1,self.J))[:,1:]
        m, n = self.X.shape
        self.m = m
        if optim_engine == 'svd':
            self.weight_finder = UnitLstsqSVD(self.X)
        elif optim_engine == 'sdr':
            self.weight_finder = UnitLstsqSDR(self.X)

    def _calculate_loss(self, g_part, weights):

        if g_part is None:

            loss = np.sum(np.dot(self.X,weights) ** 2) / self.m

            return loss

        
        g_part = np.reshape(g_part, (self.D, *(self.full_grid.shape)))

        assert g_part.shape == (self.D,*self.full_grid.shape)

        g_part = np.multiply(g_part[:,np.newaxis], self.test_function_part[np.newaxis,:,0])
        assert g_part.shape == (self.D,self.S,*(self.full_grid.shape))

        if isinstance(self.full_grid, EquiPartGrid):
            g_integrals = g_part.sum(axis=tuple(range(2, len(g_part.shape)))) * self.full_grid.get_integration_constant()
        else:    
            g_integrals = np.multiply(g_part, self.full_grid.for_integration()).sum(axis=tuple(range(2, len(g_part.shape))))
        assert g_integrals.shape == (self.D, self.S)

        y = np.reshape(g_integrals,(-1,))

        loss = np.sum((np.dot(self.X,weights) - y) ** 2) / self.m
    
        return loss



    def find_weights(self, g_part=None):

        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

        if g_part is None:

            loss, weights = self.weight_finder.solve(None,take_mean=True)

            return (loss,weights)

        else:

            g_part = np.reshape(g_part, (self.D, *(self.full_grid.shape)))

            assert g_part.shape == (self.D,*self.full_grid.shape)

            g_part = np.multiply(g_part[:,np.newaxis], self.test_function_part[np.newaxis,:,0])
            assert g_part.shape == (self.D,self.S,*(self.full_grid.shape))

            if isinstance(self.full_grid, EquiPartGrid):
                g_integrals = g_part.sum(axis=tuple(range(2, len(g_part.shape)))) * self.full_grid.get_integration_constant()
            else:    
                g_integrals = np.multiply(g_part, self.full_grid.for_integration()).sum(axis=tuple(range(2, len(g_part.shape))))
            assert g_integrals.shape == (self.D, self.S)

            y = np.reshape(g_integrals,(-1,))
             
            loss, weights = self.weight_finder.solve(y,take_mean=True)

            return (loss,weights)

           
                

        
            

class MSEWeightsFinder:

    def __init__(self, dataset, field_index, grid, dimension, order, engine, optim_engine='svd', seed=0):
        self.dataset = dataset
        self.field_index = field_index
        self.grid = grid
        self.dimension = dimension
        self.order = order
        self.seed = seed
        self.engine = engine
        self.found_weights = None

        np.random.seed(self.seed)

        #TODO: check dataset shape and grid shape

        self.D = self.dataset.shape[0]
        self.N = self.dataset.shape[1]
        self.M = self.grid.num_dims
        self.J = LinearOperator.get_vector_length(self.dimension, self.order)

        if self.field_index >= self.N:
            raise ValueError(f"There is no field with index {self.field_index}")
            
        self.derivative_dataset = np.stack([all_derivatives(dataset[d,field_index],grid,dimension,order, self.engine) for d in range(self.D)], axis=0)
        assert self.derivative_dataset.shape == (self.D,self.J,*grid.shape)

        assert self.order > 0
        self.num_lower_order = LinearOperator.get_vector_length(self.dimension, self.order-1)

        self.grid_and_fields = np.zeros([self.D,(self.M + self.N), *self.grid.shape])
        for d in range(self.D):
            self.grid_and_fields[d] = np.stack([*self.grid.by_axis(),*(self.dataset[d])],axis=0)

        derivative_part = np.moveaxis(self.derivative_dataset,1,-1)
        self.X = np.reshape(derivative_part,(-1,self.J))[:,1:]
        m, n = self.X.shape
        self.m = m
        
        if optim_engine == 'svd':
            self.weight_finder = UnitLstsqSVD(self.X)
        elif optim_engine == 'sdr':
            self.weight_finder = UnitLstsqSDR(self.X)

    def _calculate_loss(self, g_part, weights):

        if g_part is None:

            loss = np.sum(np.dot(self.X,weights) ** 2) / self.m

            return loss

        loss = np.sum((np.dot(self.X,weights) - g_part) ** 2) / self.m
    
        return loss
        


    def find_weights(self, g_part=None):

        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

        y = g_part

        loss, weights = self.weight_finder.solve(y,take_mean=True)

        return (loss,weights)



        


            
            

            
            






        
        





         
            



    

