from differential_operator import LinearOperator, Partial
from sympy import Symbol, Function, symbols
from abc import ABC, abstractmethod

class PDE(ABC):

    def __init__(self, M, N, param=None):
        self.M = M
        self.N = N


    @abstractmethod
    def get_expression(self):
        pass

    def __str__(self):
        return "\n".join([f"({L})u - {g} = 0" for L,g in self.get_expression()])




class Laplace2D(PDE):

    def __init__(self):
        super().__init__(2,1,None)

    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        g = Function('g')
        L = LinearOperator([1.0,1.0],[Partial([2,0]),Partial([0,2])])
        g = 0.0*x0 + 0.0*x1
        return [(L,g)]

        
        
L = LinearOperator([1,2,3,4],[Partial([0,1,0]),Partial([2,0,0]),Partial([1,0,1]),Partial([0,0,2])])
print(L.vectorize())

Laplace = Laplace2D()
print(Laplace)