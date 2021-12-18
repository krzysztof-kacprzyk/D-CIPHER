from abc import ABC, abstractmethod
import numpy as np


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
            return np.sin((np.pi * m * x[0])/np.sqrt(self.a)) * np.sin((np.pi * n * x[1])/np.sqrt(self.b))
        else:
            if partial.order_list == [0,0]:
                return norm_constant * np.sin((np.pi * m * x[0])/np.sqrt(self.a)) * np.sin((np.pi * n * x[1])/np.sqrt(self.b))
            elif partial.order_list == [1,0]:
                return norm_constant * (np.pi * m / self.a) * np.cos((np.pi * m * x[0])/np.sqrt(self.a)) * np.sin((np.pi * n * x[1])/np.sqrt(self.b))
            elif partial.order_list == [0,1]:
                 return norm_constant * np.sin((np.pi * m * x[0])/np.sqrt(self.a)) * (np.pi * n / self.b) * np.cos((np.pi * n * x[1])/np.sqrt(self.b))




    
