import numpy as np
from math import comb
from abc import ABC, abstractmethod
from .differential_operator import Partial, _num_combi, LinearOperator
from .libs import TVRegDiff, dxdt
from .config import get_tvdiff_params, get_trenddiff_params, get_splinediff_params, get_finitediff_params

def get_diff_engine(name):
    if name == 'numpy':
        return NumpyDiff()
    elif name == 'tv':
        return TVDiff(get_tvdiff_params())
    elif name == 'trend':
        return TrendDiff(get_trenddiff_params())
    elif name == 'spline':
        return SplineDiff(get_splinediff_params())
    elif name == 'finite':
        return FiniteDiff(get_finitediff_params())

class DerivativeEngine(ABC):

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def differentiate(self, scalar_field, grid, variable):
        pass

 
def all_derivatives(scalar_field, grid, dimension, order, engine):


    if list(scalar_field.shape) != list(grid.shape):
        raise ValueError("Scalar field and grid have different shapes")

    if dimension != grid.num_dims:
        raise ValueError("Grid has incorrect dimension")

    derivative_fields = np.zeros((LinearOperator.get_vector_length(dimension,order),*grid.shape))

    derivative_fields[0] = scalar_field
    counter = 1
    for n in range(1,order+1):
        partial = Partial([n]+([0]*(dimension-1)))
        for i in range(_num_combi(n,dimension)):
            new_order_list = partial.order_list[:]
            variable = 0
            for j in range(partial.dimension):
                if new_order_list[j] > 0:
                    new_order_list[j] -= 1
                    variable = j
                    break
            index = Partial(new_order_list).get_global_index()
            derivative_fields[counter] = engine.differentiate(derivative_fields[index],grid,variable)
            
            partial = partial.next_partial()
            counter += 1

    return derivative_fields


class NumpyDiff(DerivativeEngine):

    def __init__(self):
        super().__init__(None)

    def differentiate(self, scalar_field, grid, variable):
        coordinates = grid.axes[variable]
        return np.gradient(scalar_field, coordinates, axis=variable)
        

class TVDiff(DerivativeEngine):

    def __init__(self, params):
        super().__init__(params)
    
    def differentiate(self, scalar_field, grid, variable):
        new_scalar_field = np.moveaxis(scalar_field, variable, -1)
        org_shape = new_scalar_field.shape
        length = new_scalar_field.shape[-1]
        new_scalar = np.reshape(new_scalar_field,(-1,length))
        dx = grid.dx[0]
        derivatives = np.stack([TVRegDiff(x,dx=dx,**self.params) for x in new_scalar],axis=0)
        derivative_field = np.reshape(derivatives, org_shape)
        return np.moveaxis(derivative_field, -1, variable)

class TrendDiff(DerivativeEngine):
    def __init__(self, params):
        super().__init__(params)
    
    def differentiate(self, scalar_field, grid, variable):
        t = grid.axes[variable]
        return dxdt(scalar_field, t, kind='trend_filtered', axis=variable, **self.params)


class SplineDiff(DerivativeEngine):
    def __init__(self, params):
        super().__init__(params)

    def differentiate(self, scalar_field, grid, variable):
        t = grid.axes[variable]
        return dxdt(scalar_field, t, kind='spline', axis=variable, **self.params)

class FiniteDiff(DerivativeEngine):
    def __init__(self, params):
        super().__init__(params)

    def differentiate(self, scalar_field, grid, variable):
        t = grid.axes[variable]
        return dxdt(scalar_field, t, kind='finite_difference', axis=variable, **self.params)