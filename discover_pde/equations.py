from .differential_operator import LinearOperator, Partial
from sympy import Symbol, Function, symbols
from abc import ABC, abstractmethod

def get_pdes(name, parameters=None):
    
    if name == "TestEquation1":
        return TestEquation1()
    elif name == "Laplace2D":
        return Laplace2D()

class PDE(ABC):

    def __init__(self, param=None):
        pass
       
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    @abstractmethod
    def M(self):
        pass

    @property
    @abstractmethod
    def N(self):
        pass



    @abstractmethod
    def get_expression(self):
        pass

    @abstractmethod
    def get_solution(self, boundary_functions):
        pass

    def __str__(self):
        return "\n".join([f"({L})u - {g} = 0" for L,g in self.get_expression()])


class TestEquation1(PDE):

    def __init__(self):
        super().__init__(None)
    
    @property
    def name(self):
        return "TestEquation1"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1
    
    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        g = Function('g')
        L = LinearOperator([1.0,-1.0],[Partial([1,0]),Partial([0,1])])
        g = 0.0*x0 + 0.0*x1
        return [(L,g)]

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != 1:
            raise ValueError("Wrong number of boundary functions")
        h = boundary_functions[0]
        def func(x):
            return h(x[0] + x[1])

        return [func]
        



class Laplace2D(PDE):

    def __init__(self):
        super().__init__(None)

    @property
    def name(self):
        return "Laplace2D"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1

    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        g = Function('g')
        L = LinearOperator([1.0,1.0],[Partial([2,0]),Partial([0,2])])
        g = 0.0*x0 + 0.0*x1
        return [(L,g)]
    
    def get_solution(self, boundary_functions):
        return super().generate_solution(boundary_functions)

        
    