import time
import numpy as np
from itertools import product

from var_objective.grids import EquiPartGrid
from .differential_operator import LinearOperator
from .derivative_estimators import all_derivatives, all_derivatives_dict
from .utils.lstsq_solver import UnitLstsqLARSImproved

import pickle
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
        if optim_engine == 'lars-imp':
            self.weight_finder = UnitLstsqLARSImproved(self.X)
        else:
            raise ValueError("Only lars-imp is supported for now")

        self.bs = []

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

    # def arreq_in_list(myarr, list_arrays):
    #     return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


    def find_weights(self, g_part=None):

        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

        if g_part is None:

            loss, weights = self.weight_finder.solve(None,take_mean=True)

            return (loss,weights)

        else:

            g_part = np.reshape(g_part, (self.D, *(self.full_grid.shape)))

            assert g_part.shape == (self.D,*self.full_grid.shape)

            if isinstance(self.full_grid, EquiPartGrid):
                axes = tuple(range(1, 1 + len(self.full_grid.shape)))
                g_integrals = np.tensordot(g_part, self.test_function_part[:,0], (axes,axes)) * self.full_grid.get_integration_constant() 
            else:
                g_part = np.multiply(g_part[:,np.newaxis], self.test_function_part[np.newaxis,:,0])
                assert g_part.shape == (self.D,self.S,*(self.full_grid.shape))    
                g_integrals = np.multiply(g_part, self.full_grid.for_integration()).sum(axis=tuple(range(2, len(g_part.shape))))
            assert g_integrals.shape == (self.D, self.S)

            y = np.reshape(g_integrals,(-1,))
            # problem = {"X":self.X,"b":y}
            # id = np.random.randint(0,1000000000)
            # with open(f"results/matrices/{id}.p",'wb') as file:
            #     pickle.dump(problem,file)
            
            loss, weights = self.weight_finder.solve(y,take_mean=True)

            # if len(self.bs) < 1000:
            #     if not VariationalWeightsFinder.arreq_in_list(y,self.bs):
            #         self.bs.append(y)
            # elif len(self.bs) == 1000:
            #     print("Done")
            #     dic = {'A':self.X, 'bs':self.bs}
            #     with open('results/test_objects.p', 'wb') as f:
            #         pickle.dump(dic,f)

            return (loss,weights)

           
class VariationalWeightsFinderDictionary:

    def __init__(self,estimated_dataset, full_grid, dictQ, basis, index_limits, optim_engine='svd', seed=0):
        self.estimated_dataset = estimated_dataset
        self.full_grid = full_grid
        self.dictQ = dictQ
        self.basis = basis
        self.index_limits = index_limits
        self.seed = seed


        np.random.seed(self.seed)

        if len(self.index_limits) != self.basis.num_indexes:
            raise ValueError("Length of index_limits does not match num_indexes of basis")

        #TODO: check dataset shape and full_grid shape

        basis_indexes = list(product(*[range(1, self.index_limits[i]+1) for i in range(len(self.index_limits))]))
        self.S = len(basis_indexes)

        self.J = len(dictQ)

        self.test_function_part = np.zeros((1,self.S,self.J,*self.full_grid.shape))
        for j, ed in enumerate(dictQ):
            for s, indexes in enumerate(basis_indexes):
                self.test_function_part[0][s][j] = ed.sign() * self.basis.get_tensor(indexes, self.full_grid, ed.partial)

        self.zero_order_function_part = np.zeros((self.S,1,*self.full_grid.shape))
        for s, indexes in enumerate(basis_indexes):
            self.zero_order_function_part[s] = self.basis.get_tensor(indexes, self.full_grid, None)

        self.D = self.estimated_dataset.shape[0]
        self.N = self.estimated_dataset.shape[1]
        self.M = self.full_grid.num_dims
            
        static_part = np.zeros([self.D,self.S,self.J,*self.full_grid.shape])

        h_part = np.zeros([self.D,1,self.J,*self.full_grid.shape])
        for d in range(self.D):
            for j, ed in enumerate(dictQ):
                h_part[d][0][j] = ed.h.act(self.full_grid.by_axis(),self.estimated_dataset[d])

       
        static_part = np.multiply(self.test_function_part, h_part)
        
        if isinstance(self.full_grid, EquiPartGrid):
            integrals = static_part.sum(axis=tuple(range(3, len(static_part.shape)))) * self.full_grid.get_integration_constant()
        else:
            integrals = np.multiply(static_part, self.full_grid.for_integration()).sum(axis=tuple(range(3, len(static_part.shape))))

        assert integrals.shape == (self.D, self.S, self.J)

        self.grid_and_fields = np.zeros([self.D,(self.M + self.N), *full_grid.shape])
        for d in range(self.D):
            self.grid_and_fields[d] = np.stack([*full_grid.by_axis(),*(estimated_dataset[d])],axis=0)

        self.X = np.reshape(integrals,(-1,self.J))
        m, n = self.X.shape
        self.m = m
        if optim_engine == 'lars-imp':
            self.weight_finder = UnitLstsqLARSImproved(self.X)
        else:
            raise ValueError("Only lars-imp is supported for now")

        self.bs = []

    def _calculate_loss(self, g_part, weights, dictQ=None):

        # if dictQ is not None:
        #     weights = np.array([dictQ[i].sign()*weights[i] for i in range(len(weights))])

        if g_part is None:

            loss = np.sum(np.dot(self.X,weights) ** 2) / self.m

            return loss

        
        g_part = np.reshape(g_part, (self.D, *(self.full_grid.shape)))

        assert g_part.shape == (self.D,*self.full_grid.shape)

       

        g_part = np.multiply(g_part[:,np.newaxis], self.zero_order_function_part[np.newaxis,:,0])
        assert g_part.shape == (self.D,self.S,*(self.full_grid.shape))

        if isinstance(self.full_grid, EquiPartGrid):
            g_integrals = g_part.sum(axis=tuple(range(2, len(g_part.shape)))) * self.full_grid.get_integration_constant()
        else:    
            g_integrals = np.multiply(g_part, self.full_grid.for_integration()).sum(axis=tuple(range(2, len(g_part.shape))))
        assert g_integrals.shape == (self.D, self.S)

        y = np.reshape(g_integrals,(-1,))

        loss = np.sum((np.dot(self.X,weights) - y) ** 2) / self.m
    
        return loss

    # def arreq_in_list(myarr, list_arrays):
    #     return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


    def find_weights(self, g_part=None):

        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

        if g_part is None:

            loss, weights = self.weight_finder.solve(None,take_mean=True)

            return (loss,weights)

        else:

            g_part = np.reshape(g_part, (self.D, *(self.full_grid.shape)))

            assert g_part.shape == (self.D,*self.full_grid.shape)

            if isinstance(self.full_grid, EquiPartGrid):
                axes = tuple(range(1, 1 + len(self.full_grid.shape)))
                g_integrals = np.tensordot(g_part, self.zero_order_function_part[:,0], (axes,axes)) * self.full_grid.get_integration_constant() 
            else:
                g_part = np.multiply(g_part[:,np.newaxis], self.zero_order_function_part[np.newaxis,:,0])
                assert g_part.shape == (self.D,self.S,*(self.full_grid.shape))    
                g_integrals = np.multiply(g_part, self.full_grid.for_integration()).sum(axis=tuple(range(2, len(g_part.shape))))
            assert g_integrals.shape == (self.D, self.S)

            y = np.reshape(g_integrals,(-1,))
            # problem = {"X":self.X,"b":y}
            # id = np.random.randint(0,1000000000)
            # with open(f"results/matrices/{id}.p",'wb') as file:
            #     pickle.dump(problem,file)
            
            loss, weights = self.weight_finder.solve(y,take_mean=True)

            # if len(self.bs) < 1000:
            #     if not VariationalWeightsFinder.arreq_in_list(y,self.bs):
            #         self.bs.append(y)
            # elif len(self.bs) == 1000:
            #     print("Done")
            #     dic = {'A':self.X, 'bs':self.bs}
            #     with open('results/test_objects.p', 'wb') as f:
            #         pickle.dump(dic,f)

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
        
        if optim_engine == 'lars-imp':
            self.weight_finder = UnitLstsqLARSImproved(self.X)
        else:
            raise ValueError("Only lars-imp is supported for now")


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


class MSEWeightsFinderDictionary:

    def __init__(self, dataset, grid, dictQ, engine, optim_engine='svd', seed=0):
        self.dataset = dataset
        self.grid = grid
        self.dictQ = dictQ
        self.seed = seed
        self.engine = engine
        self.found_weights = None

        np.random.seed(self.seed)

        #TODO: check dataset shape and grid shape

        self.D = self.dataset.shape[0]
        self.N = self.dataset.shape[1]
        self.M = self.grid.num_dims
        self.J = len(dictQ)
   
        self.derivative_dataset = np.stack([all_derivatives_dict(dataset[d],grid,dictQ, self.engine) for d in range(self.D)], axis=0)
        assert self.derivative_dataset.shape == (self.D,self.J,*grid.shape)

        self.grid_and_fields = np.zeros([self.D,(self.M + self.N), *self.grid.shape])
        for d in range(self.D):
            self.grid_and_fields[d] = np.stack([*self.grid.by_axis(),*(self.dataset[d])],axis=0)

        derivative_part = np.moveaxis(self.derivative_dataset,1,-1)
        self.X = np.reshape(derivative_part,(-1,self.J))[:,:]
        m, n = self.X.shape
        self.m = m
        
        if optim_engine == 'lars-imp':
            self.weight_finder = UnitLstsqLARSImproved(self.X)
        else:
            raise ValueError("Only lars-imp is supported for now")


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



        


            
            

            
            






        
        





         
            



    

