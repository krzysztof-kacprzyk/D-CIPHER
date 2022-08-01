import numpy as np
from scipy.special import comb
from abc import ABC, abstractmethod
from .differential_operator import Partial, _num_combi, LinearOperator
from .libs import TVRegDiff, dxdt
from .config import get_tvdiff_params, get_trenddiff_params, get_splinediff_params, get_finitediff_params
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def get_diff_engine(name,seed=0):
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
    elif name == 'gp':
        return GPDiff({'seed':seed, 'delta':1e-3})
    elif name == 'gp2':
        return GP2Diff({'seed':seed, 'delta':1e-3})

class DerivativeEngine(ABC):

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def differentiate(self, scalar_field, grid, variable):
        pass

 
def all_derivatives(scalar_field, grid, dimension, order, engine):

    if isinstance(engine, GP2Diff):
        return engine.finite_diff(scalar_field, grid, dimension, order, engine)


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

def all_derivatives_dict(vector_field, grid, dictQ, engine):

    computed_eds = []
    for j, ed in enumerate(dictQ):
        partial = ed.partial
        transformed = ed.h.act(grid.by_axis(),vector_field)
        new_order_list = partial.order_list[:]
        for k in range(partial.dimension):
            while new_order_list[k] > 0:
                new_order_list[k] -= 1
                variable = k
                transformed = engine.differentiate(transformed,grid,variable)
        computed_eds.append(transformed)
    
    return np.stack(computed_eds,axis=0)



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

class GPDiff(DerivativeEngine):

    def __init__(self, params={'seed':0}):
        super().__init__(params)
        # Configure Gaussian Process
        kernel = ConstantKernel()*RBF() + WhiteKernel()
        # kernel = RBF() + WhiteKernel()
        self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=self.params['seed'])

    def differentiate(self, scalar_field, grid, variable):
        u_obs = scalar_field.flatten()
        mean = np.mean(u_obs)
        std = np.std(u_obs)
        org_cov = grid.as_covariates()
        self.gpr.fit(org_cov, (u_obs - mean)/std)

        delta_t = self.params['delta']
        diff_vector = np.zeros(org_cov.shape[1])
        diff_vector[variable] = delta_t
        forward_cov = org_cov + diff_vector
        backward_cov = org_cov - diff_vector

        forward_values = self.gpr.predict(forward_cov) * std + mean
        backward_values = self.gpr.predict(backward_cov) * std + mean

        derivative = (forward_values - backward_values) / (2*delta_t)

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # # Plot the surface.
        # surf = ax.plot_surface(grid.by_axis()[0], grid.by_axis()[1], scalar_field, linewidth=0, antialiased=False)
        # plt.show()

        # surf = ax.plot_surface(grid.by_axis()[0], grid.by_axis()[1], derivative.reshape(grid.shape), linewidth=0, antialiased=False)
        # plt.show()
        return derivative.reshape(grid.shape)


class Stencil:

    def __init__(self,directions, coefficients, denominator):
        self.directions = directions
        self.coefficients = coefficients
        self.denominator = denominator

class GP2Diff(DerivativeEngine):

    def __init__(self, params={'seed':0}):
        super().__init__(params)
        # Configure Gaussian Process
        kernel = ConstantKernel()*RBF() + WhiteKernel()
        self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=self.params['seed'])

    def finite_diff(self,scalar_field, grid, dimension, order, engine):

        if (order > 2) or (dimension > 2):
            raise ValueError(f"Not currently implemented for order {order} and dimension {dimension}")


        if list(scalar_field.shape) != list(grid.shape):
            raise ValueError("Scalar field and grid have different shapes")

        if dimension != grid.num_dims:
            raise ValueError("Grid has incorrect dimension")

        derivative_fields = np.zeros((LinearOperator.get_vector_length(dimension,order),*grid.shape))

        derivative_fields[0] = scalar_field

        u_obs = scalar_field.flatten()
        mean = np.mean(u_obs)
        std = np.std(u_obs)
        org_cov = grid.as_covariates()
        self.gpr.fit(org_cov, (u_obs - mean)/std)
        stencils = {
            (1):Stencil([[1],[-1]],[1,-1],lambda t: 2*t),
            (2):Stencil([[1],[0],[-1]],[1,-2,1],lambda t: t**2),
            (1,0):Stencil([[1,0],[-1,0]],[1,-1], lambda t: 2*t),
            (0,1):Stencil([[0,1],[0,-1]],[1,-1], lambda t: 2*t),
            (1,1):Stencil([[1,1],[1,-1],[-1,1],[-1,-1]],[1,-1,-1,1], lambda t: 4*t**2),
            (2,0):Stencil([[1,0],[0,0],[-1,0]],[1,-2,1], lambda t: t**2),
            (0,2):Stencil([[0,1],[0,0],[0,-1]],[1,-2,1], lambda t: t**2)
        }

        counter = 1
        for n in range(1,order+1):
            partial = Partial([n]+([0]*(dimension-1)))

            for i in range(_num_combi(n,dimension)):

                stencil = stencils[tuple(partial.order_list)]
                der_field = self.differentiate(org_cov,grid,stencil, std, mean)
                derivative_fields[counter] = der_field
                
                partial = partial.next_partial()
                counter += 1

        return derivative_fields


    def differentiate(self, org_cov, grid, stencil, std, mean):
        

        delta_t = self.params['delta']
        directions = stencil.directions
        coefficients = stencil.coefficients
        denom = stencil.denominator(delta_t)

        terms = []
        for i in range(len(directions)):
            diff_vector = np.array(directions[i]) * delta_t
            term_argument = org_cov + diff_vector
            term_values = self.gpr.predict(term_argument) * std + mean
            terms.append(term_values)

        derivative = np.zeros(org_cov.shape[0])
        for coeff, term in zip(coefficients,terms):
            derivative += coeff * term
    
        derivative /= denom
        

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # # Plot the surface.
        # surf = ax.plot_surface(grid.by_axis()[0], grid.by_axis()[1], scalar_field, linewidth=0, antialiased=False)
        # plt.show()

        # surf = ax.plot_surface(grid.by_axis()[0], grid.by_axis()[1], derivative.reshape(grid.shape), linewidth=0, antialiased=False)
        # plt.show()
        return derivative.reshape(grid.shape)