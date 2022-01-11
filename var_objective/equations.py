from multiprocessing import Value
from .differential_operator import LinearOperator, Partial
from .population_models import SLM
from .heat_equation import HeatEquationNeumann1D
from sympy import Symbol, Function, symbols, sin, exp, pi
from abc import ABC, abstractmethod
import numpy as np

def get_pdes(name, parameters=None):
    
    if name == "TestEquation1":
        return TestEquation1()
    elif name == "TestEquation2":
        return TestEquation2()
    elif name == "Laplace2D":
        return Laplace2D()
    elif name == "SLM1":
        return SLM1()
    elif name == "HeatEquation_0.1":
        return HeatEquation1(0.1)
    else:
        raise ValueError(f"Unknown equation: {name}")

class PDE(ABC):

    def __init__(self, params=None):
        self.params = params

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

    @property
    @abstractmethod
    def num_conditions(self):
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

    @property
    def num_conditions(self):
        return 1
    
    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        g = Function('g')
        L = LinearOperator([1.0,-1.0],[Partial([1,0]),Partial([0,1])])
        g = 0.0*x0 + 0.0*x1
        return [(L,g)]

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        h = boundary_functions[0]
        def func(grid):
            x = grid.by_axis()
            return h(x[0] + x[1])

        return [func]
        

class TestEquation2(PDE):
    
    def __init__(self):
        super().__init__(None)
    
    @property
    def name(self):
        return "TestEquation2"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1

    @property
    def num_conditions(self):
        return 1
    
    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        g = Function('g')
        L = LinearOperator([1.0,2.0],[Partial([1,0]),Partial([0,1])])
        g = 0.0*x0 + sin(x1)
        return [(L,g)]

    def get_solution(self, boundary_functions):
        """
            boundary_function h specifies the boundary condition u0(t,0) = h(t)
        """
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        h = boundary_functions[0]
        def func(grid):
            x = grid.by_axis()
            return -0.5 * np.cos(x[1]) + h(x[0] - x[1]/2) + 0.5

        return [func]


class SLM1(PDE):
    
    def __init__(self):
        super().__init__(None)
    
    @property
    def name(self):
        return "SLM1"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1

    @property
    def num_conditions(self):
        return 2
    
    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g = Function('g')
        L = LinearOperator([1.0,1.0],[Partial([1,0]),Partial([0,1])])
        g = 2*exp(x1 - 1)*u0
        return [(L,g)]

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        death_rate = lambda x: 2*np.exp((x-1))
        birth_rate = boundary_functions[0]
        initial_age_distribution = boundary_functions[1]

        def func(grid):
            assert grid.num_dims == 2
            axes = grid.axes
            widths = grid.widths
            delta_t = 0.001
            slm = SLM(death_rate,birth_rate,initial_age_distribution)
            U = slm.solve_second_order(widths[0], delta_t, widths[1])
            sol = np.zeros(grid.shape)
            grid_trans = grid.as_grid()
            for t, g_t in enumerate(grid_trans):
                for a, g_t_a in enumerate(g_t):
                    ind_t = int(g_t_a[0] / delta_t)
                    ind_a = int(g_t_a[1] / delta_t)
                    sol[t,a] = U[ind_t, ind_a]

            return sol

        return [func]


class HeatEquation1(PDE):

    def __init__(self, k):
        super().__init__({'k': k})

    @property
    def name(self):
        return "HeatEquation1"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1

    @property
    def num_conditions(self):
        return 1

    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        g = Function('g')
        L = LinearOperator([1.0,-self.params['k']],[Partial([1,0]),Partial([0,2])])
        g = 4.0*sin(2*x1 * pi)
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        heat_source = lambda X: 4*np.sin(2*np.pi*X[1])
        boundary1  = lambda x: np.zeros_like(x)
        boundary2 = lambda x: np.zeros_like(x)
        initial_temp = boundary_functions[0]

        def func(grid):
            assert grid.num_dims == 2
            axes = grid.axes
            widths = grid.widths
            delta_t = 0.001
            delta_x = 0.001
            heat_equation = HeatEquationNeumann1D(self.params['k'], heat_source, boundary1, boundary2, initial_temp)
            U = heat_equation.btcs(widths[0], widths[1], delta_t, delta_x)
            sol = np.zeros(grid.shape)
            grid_trans = grid.as_grid()
            for t, g_t in enumerate(grid_trans):
                for x, g_t_x in enumerate(g_t):
                    ind_t = int(g_t_x[0] / delta_t)
                    ind_x = int(g_t_x[1] / delta_x)
                    sol[t,x] = U[ind_t, ind_x]

            return sol

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

        
    