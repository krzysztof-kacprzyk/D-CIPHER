from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import BSpline

from var_objective.differential_operator import Partial

from .grids import EquiPartGrid
import matplotlib.pyplot as plt


class BasisFunction(ABC):

    def __init__(self, widths):
        self.widths = widths

    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def max_order(self):
        pass

    @property
    @abstractmethod
    def num_indexes(self):
        pass

    def _verify(self, indexes, num_variables, partial=None):

        assert len(indexes) == self.num_indexes
        assert num_variables == self.dimension
        if partial != None:
            assert partial.dimension == self.dimension
            assert partial.order <= self.max_order
    
    def get_tensor(self, indexes, grid, partial=None):
        pass


class BSplineFreq2D(BasisFunction):

    def __init__(self, widths, order):

        assert len(widths) == 2

        self.a = widths[0]
        self.b = widths[1]

        self.order = order

    @property
    def dimension(self):
        return 2
    
    @property
    def max_order(self):
        return self.order
    
    @property
    def num_indexes(self):
        return 2
    
    def get_tensor(self, indexes, grid, partial=None):

        x = grid.by_axis()
        self._verify(indexes, x.shape[0], partial)

        m = indexes[0]
        n = indexes[1]

        end1 = self.a / m
        end2 = self.b / n

        
        # We need order+1 because we want order-th derivative to be continuous
        b1 = BSpline.basis_element(np.linspace(0, end1, self.order+1+2),extrapolate='periodic')
        b2 = BSpline.basis_element(np.linspace(0, end2, self.order+1+2),extrapolate='periodic')

        axes = grid.axes
        x1 = axes[0]
        x2 = axes[1]

        if partial != None:

            o1 = partial.order_list[0]
            for i in range(o1):
                b1 = b1.derivative()
            
            o2 = partial.order_list[1]
            for i in range(o2):
                b2 = b2.derivative()
        
        return np.outer(b1(x1), b2(x2))

        

class FourierSine2D(BasisFunction):
    
    def __init__(self, widths):

        assert len(widths) == 2

        self.a = widths[0]
        self.b = widths[1]

    @property
    def dimension(self):
        return 2
    
    @property
    def max_order(self):
        return 1
    
    @property
    def num_indexes(self):
        return 2
    
    def get_tensor(self, indexes, grid, partial=None):

        x = grid.by_axis()
        self._verify(indexes, x.shape[0], partial)

        m = indexes[0]
        n = indexes[1]

        norm_constant = 2/np.sqrt(self.a * self.b)
        
        if partial == None:
            return norm_constant * np.sin((np.pi * m * x[0])/self.a) * np.sin((np.pi * n * x[1])/self.b)
        else:
            if partial.order_list == [0,0]:
                return norm_constant * np.sin((np.pi * m * x[0])/self.a) * np.sin((np.pi * n * x[1])/self.b)
            elif partial.order_list == [1,0]:
                return norm_constant * (np.pi * m / self.a) * np.cos((np.pi * m * x[0])/self.a) * np.sin((np.pi * n * x[1])/self.b)
            elif partial.order_list == [0,1]:
                 return norm_constant * np.sin((np.pi * m * x[0])/self.a) * (np.pi * n / self.b) * np.cos((np.pi * n * x[1])/self.b)


class Fake(BasisFunction):
    
    def __init__(self, widths):

        assert len(widths) == 2

        self.a = widths[0]
        self.b = widths[1]

    @property
    def dimension(self):
        return 2
    
    @property
    def max_order(self):
        return 2
    
    @property
    def num_indexes(self):
        return 2
    
    def get_tensor(self, indexes, grid, partial=None):

        x = grid.by_axis()
        self._verify(indexes, x.shape[0], partial)

        m = indexes[0]
        n = indexes[1]

        norm_constant = 2/np.sqrt(self.a * self.b)
        
        return norm_constant * np.sin((np.pi * m * x[0])/self.a) * np.sin((np.pi * n * x[1])/self.b)

    
if __name__ == "__main__":

    observed_grid = EquiPartGrid([1.0, 1.0], 500)


    basis = BSplineFreq2D([1.0, 1.0], 2)
    p1 = Partial([1,0])
    field = basis.get_tensor([2,2], observed_grid, partial=p1)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(observed_grid.by_axis()[0], observed_grid.by_axis()[1], field, linewidth=0, antialiased=False)


    plt.show()