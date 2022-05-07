from multiprocessing import Value

from var_objective.flow import get_flow_potential_2D
from var_objective.wave_equation import DampedWaveEquationDirichlet1D, WaveEquationDirichlet1D

from .coulomb import get_potential_2D, get_potential_3D
from .differential_operator import LinearOperator, Partial
from .population_models import SLM
from .heat_equation import HeatEquationNeumann1D
from sympy import Symbol, Function, symbols, sin, exp, pi, lambdify
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

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
    elif name == "HeatEquation_1.0":
        return HeatEquation1(1.0)
    elif name == "HeatEquation2":
        return HeatEquation2(0.2)
    elif name == "HeatEquation2_L1":
        return HeatEquation2_L1(0.2)
    elif name == "Coulomb2D":
        return Coulomb2D(1.0)
    elif name == "Coulomb3D":
        return Coulomb3D(1.0)
    elif name == "Flow2D":
        return Flow2D()
    elif name == "HeatEquation3_L1":
        return HeatEquation3_L1(0.2,1.8)
    elif name == "HeatEquation4_L1":
        return HeatEquation4_L1(0.2,1.8)
    elif name == "WaveEquation1_L1":
        return WaveEquation1_L1(2)
    elif name == "WaveEquation2_L1":
        return WaveEquation2_L1(3,0.5)
    elif name == "HarmonicOscillator":
        return HarmonicOscillator(3.0)
    elif name == "DampedHarmonicOscillator":
        return DampedHarmonicOscillator(4.0,0.5)
    elif name == "DrivenHarmonicOscillator":
        return DrivenHarmonicOscillator(4.0,0.5,3.0,5.0)
    elif name == "HeatEquation5_L1":
        return HeatEquation3_L1(0.25,1.8)
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

    @abstractmethod
    def get_functional_form_normalized(self,norm='l2'):
        pass

    def __str__(self):
        return "\n".join([f"({L})u - {g} = 0" for L,g in self.get_expression()])

    def get_expression_normalized(self,norm='l2'):
        equations = self.get_expression()
        new_equations = []
        for equation in equations:
            L, g  = equation
            length = L.get_length(norm=norm)
            L = L.normalize(norm=norm)
            g = g / length
            new_equations.append((L,g))
        return new_equations

    def get_independent_variables(self):
        variables = []
        for i in range(self.M):
            xi = symbols(f'x{i}',real=True)
            variables.append(xi)
        return variables
    def get_dependent_variables(self):
        variables = []
        for i in range(self.N):
            ui = symbols(f'u{i}',real=True)
            variables.append(ui)
        return variables
    def get_all_variables(self):
        return self.get_independent_variables() + self.get_dependent_variables()

    
    def numpify_g(self, g_symb):
        variables = self.get_all_variables()
        g_numpy = lambdify(variables, g_symb, 'numpy')
        return g_numpy


        
    
        






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
        
class HeatEquation2(PDE):

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
        g = np.sqrt(1.04) * x0 * x1
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        heat_source = lambda X: np.sqrt(1.04) * X[0] * X[1]
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

class HeatEquation2_L1(PDE):

    def __init__(self, k):
        super().__init__({'k': k})

    @property
    def name(self):
        return "HeatEquation2_L1"

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
        g = (1+np.abs(self.params['k'])) * x0 * x1
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        heat_source = lambda X: (1+np.abs(self.params['k'])) * X[0] * X[1]
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

class HeatEquation3_L1(PDE):

    def __init__(self, k, theta):
        super().__init__({'k': k, 'theta':theta})

    @property
    def name(self):
        return "HeatEquation3_L1"

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
        g = (1+np.abs(self.params['k'])) * exp(self.params['theta']*x0)
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        heat_source = lambda X: (1+np.abs(self.params['k'])) * np.exp(self.params['theta']*X[0])
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

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = exp(C*X0)

        return [g]
    
class HeatEquation4_L1(PDE):

    def __init__(self, k, theta):
        super().__init__({'k': k, 'theta':theta})

    @property
    def name(self):
        return "HeatEquation4_L1"

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
        g = (1+np.abs(self.params['k'])) * x1 * exp(self.params['theta']*x0)
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        heat_source = lambda X: (1+np.abs(self.params['k'])) * X[1] * np.exp(self.params['theta']*X[0])
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

class WaveEquation1_L1(PDE):

    def __init__(self, k):
        super().__init__({'k': k})

    @property
    def name(self):
        return "WaveEquation1_L1"

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
        L = LinearOperator([1.0,-self.params['k']**2],[Partial([2,0]),Partial([0,2])])
        g = 0*x0+0*x1
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        wave_source = lambda X: np.zeros_like(X[0])
        initial_wave = boundary_functions[0]

        def func(grid):
            assert grid.num_dims == 2
            axes = grid.axes
            widths = grid.widths
            delta_t = 0.001
            delta_x = 0.001
            wave_equation = WaveEquationDirichlet1D(self.params['k'],wave_source,initial_wave)
            U = wave_equation.idm(widths[0], widths[1], delta_t, delta_x)
            sol = np.zeros(grid.shape)
            grid_trans = grid.as_grid()
            for t, g_t in enumerate(grid_trans):
                for x, g_t_x in enumerate(g_t):
                    ind_t = int(g_t_x[0] / delta_t)
                    ind_x = int(g_t_x[1] / delta_x)
                    sol[t,x] = U[ind_t, ind_x]

            return sol

        return [func]

class WaveEquation2_L1(PDE):

    def __init__(self, k, d):
        super().__init__({'k': k, 'd':d})

    @property
    def name(self):
        return "WaveEquation2_L1"

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
        L = LinearOperator([0.5,-self.params['k']**2,self.params['d']],[Partial([2,0]),Partial([0,2]),Partial([1,0])])
        g = 0*x0+0*x1
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        wave_source = lambda X: np.zeros_like(X[0])
        initial_wave = boundary_functions[0]

        def func(grid):
            assert grid.num_dims == 2
            axes = grid.axes
            widths = grid.widths
            delta_t = 0.001
            delta_x = 0.001
            wave_equation = DampedWaveEquationDirichlet1D(self.params['k'],self.params['d']/2,wave_source,initial_wave)
            U = wave_equation.idm(widths[0], widths[1], delta_t, delta_x)
            sol = np.zeros(grid.shape)
            grid_trans = grid.as_grid()
            for t, g_t in enumerate(grid_trans):
                for x, g_t_x in enumerate(g_t):
                    ind_t = int(g_t_x[0] / delta_t)
                    ind_x = int(g_t_x[1] / delta_x)
                    sol[t,x] = U[ind_t, ind_x]

            return sol

        return [func]


class Coulomb2D(PDE):

    def __init__(self,k):
        super().__init__({'k':k})
    
    @property
    def name(self):
        return "Coulomb2D"

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
        g = Function('g')
        L = LinearOperator([1.0,1.0],[Partial([2,0]),Partial([0,2])])
        g = 0.0*x0 + 0.0*x1
        return [(L,g)]

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
    
        def func(grid):
            locs = boundary_functions[0]
            charges = boundary_functions[1]
            locs[:,0] *= grid.widths[0]
            locs[:,1] *= grid.widths[1]
            return get_potential_2D(grid,locs,charges,self.params['k'])

        return [func]

class Flow2D(PDE):

    def __init__(self):
        super().__init__(None)
    
    @property
    def name(self):
        return "Flow2D"

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
        g = Function('g')
        L = LinearOperator([1.0,1.0],[Partial([2,0]),Partial([0,2])])
        g = 0.0*x0 + 0.0*x1
        return [(L,g)]

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
    
        def func(grid):
            locs = boundary_functions[0]
            strengths = boundary_functions[1]
            locs[:,0] *= grid.widths[0]
            locs[:,1] *= grid.widths[1]
            return get_flow_potential_2D(grid,locs,strengths)

        return [func]

class Coulomb3D(PDE):

    def __init__(self,k):
        super().__init__({'k':k})
    
    @property
    def name(self):
        return "Coulomb3D"

    @property
    def M(self):
        return 3

    @property
    def N(self):
        return  1

    @property
    def num_conditions(self):
        return 2 # positions and charges
    
    def get_expression(self):
        x0,x1,x2 = symbols('x0,x1,x2', real=True)
        g = Function('g')
        L = LinearOperator([1.0,1.0,1.0],[Partial([2,0,0]),Partial([0,2,0]),Partial([0,0,2])])
        g = 0.0*x0 + 0.0*x1 + 0.0*x2
        return [(L,g)]

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
    
        def func(grid):
            locs = boundary_functions[0]
            charges = boundary_functions[1]
            locs[:,0] *= grid.widths[0]
            locs[:,1] *= grid.widths[1]
            locs[:,2] *= grid.widths[2]
            
            return get_potential_3D(grid,locs,charges,self.params['k'])

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

class HarmonicOscillator(PDE):
    
    def __init__(self, k):
        super().__init__(params={'k':k})
    
    @property
    def name(self):
        return "HarmonicOsciallator"

    @property
    def M(self):
        return 1

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
        L = LinearOperator([1.0],[Partial([2])])
        g = -(self.params['k']**2)*u0 + 0.0*x0
        return [(L,g)]

    def get_solution(self, boundary_functions):
        
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        x0 = boundary_functions[0]
        dxdt0 = boundary_functions[1]
        def func(grid):
            x = grid.by_axis()[0]

            return x0*np.cos(self.params['k']*x) + (dxdt0 / self.params['k']) * np.sin(self.params['k']*x)

        return [func]

class DampedHarmonicOscillator(PDE):
    
    def __init__(self, k, d):
        super().__init__(params={'k':k,'d':d})
    
    @property
    def name(self):
        return "DampedHarmonicOsciallator"

    @property
    def M(self):
        return 1

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
        L = LinearOperator([2*self.params['k']*self.params['d'],1.0],[Partial([1]),Partial([2])])
        g = -(self.params['k']**2)*u0 + 0.0*x0
        return [(L,g)]

    def get_solution(self, boundary_functions):
        
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        x0 = boundary_functions[0]
        dxdt0 = boundary_functions[1]
        A = x0
        B = ((dxdt0 / self.params['k']) + A*self.params['d']) / np.sqrt(1-self.params['d']**2)
        def func(grid):
            x = grid.by_axis()[0]
            return np.exp(-self.params['d']*self.params['k']*x)*(A*np.cos(self.params['k']*np.sqrt(1-self.params['d']**2)*x) + B*np.sin(self.params['k']*np.sqrt(1-self.params['d']**2)*x))

        return [func]

class DrivenHarmonicOscillator(PDE):
    
    def __init__(self, k, d, k_r, F):
        super().__init__(params={'k':k,'d':d, 'k_r':k_r, 'F':F})
    
    @property
    def name(self):
        return "DrivenHarmonicOsciallator"

    @property
    def M(self):
        return 1

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
        L = LinearOperator([2*self.params['k']*self.params['d'],1.0],[Partial([1]),Partial([2])])
        g = -(self.params['k']**2)*u0 + self.params['F']*sin(self.params['k_r']*x0)
        return [(L,g)]

    def get_solution(self, boundary_functions):
        
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        x0 = boundary_functions[0]
        dxdt0 = boundary_functions[1]

        denom = (2*self.params['d']*self.params['k']*self.params['k_r']) ** 2 + (self.params['k'] ** 2 - self.params['k_r'] ** 2) ** 2
        partA =  - self.params['F'] * (2*self.params['d']*self.params['k']*self.params['k_r']) / denom
        partB = self.params['F'] * (self.params['k'] ** 2 - self.params['k_r'] ** 2) / denom
        A = x0 - partA
        B = (dxdt0 - partB*self.params['k_r'] + A*self.params['d']*self.params['k']) / (self.params['k'] * np.sqrt(1-self.params['d']**2))
        def func(grid):
            x = grid.by_axis()[0]
            return np.exp(-self.params['d']*self.params['k']*x) \
                *(A*np.cos(self.params['k']*np.sqrt(1-self.params['d']**2)*x) \
                + B*np.sin(self.params['k']*np.sqrt(1-self.params['d']**2)*x)) \
                + partA * np.cos(self.params['k_r']*x) \
                + partB * np.sin(self.params['k_r']*x)

        return [func]

    def get_functional_form_normalized(self,norm='l2'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        C = Symbol('C', real=True, positive=True)
      
        L = LinearOperator([2*self.params['k']*self.params['d'],1.0],[Partial([1]),Partial([2])])
        length = L.get_length(norm=norm)
        if self.params['k']**2 / length == 1.0:
            param1 = 1.0
        else:
            param1 = C
        
        if self.params['F'] / length == 1.0:
            param2 = 1.0
        else:
            param2 = C
        g = -param1*X1 + param2*sin(C*X0)
        return [g]


        
    
if __name__ == '__main__':

    eq = TestEquation2()

    g_part = eq.get_g_numpy(0,normalized=True)

    t = np.linspace(0,2*np.pi,1000)

    plt.plot(t,g_part(t,t,t))
    plt.show()


