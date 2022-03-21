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

        self.norm_dict = {0:1/3, 1:11/60, 2:151/1260, 3:15619/181440, 4:655177/9979200}

        if order > 4:
            raise ValueError("Order can be at most 4 unless you provide normalization constants")

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

        norm = np.sqrt(self.norm_dict[self.order] ** 2 * self.a * self.b)
       
        
        return np.outer(b1(x1), b2(x2)) / norm

class BSplineTrans2D(BasisFunction):

    def __init__(self, widths, order, max_indexes):

        assert len(widths) == 2

        self.a = widths[0]
        self.b = widths[1]

        self.order = order

        self.norm_dict = {0:1/3, 1:11/60, 2:151/1260, 3:15619/181440, 4:655177/9979200}

        if order > 4:
            raise ValueError("Order can be at most 4 unless you provide normalization constants")

        self.max_indexes = max_indexes

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

        m_max = self.max_indexes[0]
        n_max = self.max_indexes[1]

        width1 = (self.a / m_max)
        start1 = width1 * (m-1)
        end1 = width1 * m

        width2 = (self.b / n_max)
        start2 = width2 * (n-1)
        end2 = width2 * n

        
        # We need order+1 because we want order-th derivative to be continuous
        b1 = BSpline.basis_element(np.linspace(start1, end1, self.order+1+2),extrapolate=False)
        b2 = BSpline.basis_element(np.linspace(start2, end2, self.order+1+2),extrapolate=False)

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

        b1_applied = b1(x1)
        b1_applied[np.isnan(b1_applied)] = 0.0

        b2_applied = b2(x2)
        b2_applied[np.isnan(b2_applied)] = 0.0

        norm = np.sqrt(self.norm_dict[self.order] ** 2 * width1 * width2)
       
        
        return np.outer(b1_applied, b2_applied) / norm

        

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

    observed_grid = EquiPartGrid([2.0, 4.0], 500)


    basis = BSplineTrans2D([2.0, 4.0], 2, (5,5))
    p1 = Partial([0,0])
    field = basis.get_tensor([3,2], observed_grid, partial=p1)

    print(np.sum(field ** 2) * (8.0 / (500 ** 2)))


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(observed_grid.by_axis()[0], observed_grid.by_axis()[1], field, linewidth=0, antialiased=False)


    plt.show()