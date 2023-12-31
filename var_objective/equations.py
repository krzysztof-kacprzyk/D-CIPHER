from multiprocessing import Value

from matplotlib import projections
from var_objective.simulators.burger import Burger

from var_objective.simulators.flow import get_complex_potential_im, get_complex_potential_re, get_flow_potential_2D, get_flow_stream_2D
from var_objective.simulators.wave_equation import DampedWaveEquationDirichlet1D, WaveEquationDirichlet1D

from var_objective.simulators.coulomb import get_potential_2D, get_potential_3D
from .differential_operator import ED, EFunction, LinearOperator, Partial, proj_0, square_0, proj_1, cubic_0
from var_objective.simulators.population_models import SLM
from var_objective.simulators.heat_equation import HeatEquationNeumann1D
from sympy import Symbol, Function, symbols, sin, exp, pi, lambdify
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
        return WaveEquation1_L1(1.0)
    elif name == "WaveEquation2_L1":
        return WaveEquation2_L1(1.0,2.0)
    elif name == "WaveEquation3_L1":
        return WaveEquation3_L1(1.0)
    elif name == "HarmonicOscillator":
        return HarmonicOscillator(3.0)
    elif name == "DampedHarmonicOscillator":
        return DampedHarmonicOscillator(4.0,0.5)
    elif name == "DrivenHarmonicOscillator":
        return DrivenHarmonicOscillator(4.0,0.5,3.0,5.0)
    elif name == "HeatEquation5_L1":
        return HeatEquation3_L1(0.25,1.8)
    elif name == "Liouville_L1":
        return Liouville_L1(0.8)
    elif name == "Liouville2_L1":
        return Liouville2_L1(1.6)
    elif name == "SLM1Dict":
        return SLM1Dict()
    elif name == "SLM1DictMany":
        return SLM1DictMany()
    elif name == "BurgerDict":
        return BurgerDict(v=0.2)
    elif name == "KdVDict":
        return KdVDict()
    elif name == "KSDict":
        return KSDict()
    elif name == "FullFlow2D":
        return FullFlow2D()
    elif name == "HeatEquationHomo":
        return HeatEquationHomo(0.25)
    elif name == "NS":
        return NS()
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
        return 1
    
    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g = Function('g')
        L = LinearOperator([1.0,1.0],[Partial([1,0]),Partial([0,1])])
        g = 2*exp(1.5*x1)*u0
        return [(L,g)]

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        death_rate = lambda x: 2*np.exp(1.5*x)
        birth_rate = lambda x: np.where(x < 1,np.sin(x*np.pi),np.zeros_like(x))
        initial_age_distribution = boundary_functions[0]

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

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = exp(C*X1)*X2

        return [g]

        
class SLM1Dict(PDE):
    
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
        return 1

    def get_dictionaries(self):
        dict0 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([1,1]),proj_0)
        ]
        return [dict0]
    def get_weights(self,normalize=False):
        weights0 = np.array([1.0,1.0,0.0,0.0,0.0])
        if normalize:
            weights0 /= np.linalg.norm(weights0,1)
        return [weights0]

    def get_free_parts(self,normalize=False):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g0 = Function('g')
        g0 = -2*exp(1.5*x1)*u0
        if normalize:
            g0 = g0 / 2
        return [g0]

    def get_expression(self):
        return super().get_expression()
    

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        death_rate = lambda x: 2*np.exp(1.5*x)
        birth_rate = lambda x: np.where(x < 1,np.sin(x*np.pi),np.zeros_like(x))
        initial_age_distribution = boundary_functions[0]

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

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = -exp(C*X1)*X2

        return [g]

class SLM1DictMany(PDE):
    
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
        return 1

    def get_dictionaries(self):
        dict0 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
        ]
        dict1 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
        ]
        dict2 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
        ]
        dict3 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([1,1]),proj_0)
        ]
        dict4 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([1,1]),proj_0),
            ED(Partial([0,1]),square_0)
        ]
        dict5 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([1,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),square_0)
        ]
        dict6 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([1,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),square_0),
            ED(Partial([1,0]),square_0)
        ]
        dict7 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([1,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),square_0),
            ED(Partial([1,0]),square_0),
            ED(Partial([2,0]),square_0)
        ]
        dict8 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([1,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),square_0),
            ED(Partial([1,0]),square_0),
            ED(Partial([2,0]),square_0),
            ED(Partial([1,0]),cubic_0)
        ]
        dict9 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([1,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),square_0),
            ED(Partial([1,0]),square_0),
            ED(Partial([2,0]),square_0),
            ED(Partial([1,0]),cubic_0),
            ED(Partial([0,1]),cubic_0)
        ]

        
        return [dict0,dict1,dict2,dict3,dict4,dict5,dict6,dict7,dict8,dict9]
    def get_weights(self,normalize=False):
        weights0 = np.array([1.0,1.0])
        weights1 = np.array([1.0,1.0,0.0])
        weights2 = np.array([1.0,1.0,0.0,0.0])
        weights3 = np.array([1.0,1.0,0.0,0.0,0.0])
        weights4 = np.array([1.0,1.0,0.0,0.0,0.0,0.0])
        weights5 = np.array([1.0,1.0,0.0,0.0,0.0,0.0,0.0])
        weights6 = np.array([1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0])
        weights7 = np.array([1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        weights8 = np.array([1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        weights9 = np.array([1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        if normalize:
            weights0 /= np.linalg.norm(weights0,1)
            weights1 /= np.linalg.norm(weights1,1)
            weights2 /= np.linalg.norm(weights2,1)
            weights3 /= np.linalg.norm(weights3,1)
            weights4 /= np.linalg.norm(weights4,1)
            weights5 /= np.linalg.norm(weights5,1)
            weights6 /= np.linalg.norm(weights6,1)
            weights7 /= np.linalg.norm(weights7,1)
            weights8 /= np.linalg.norm(weights8,1)
            weights9 /= np.linalg.norm(weights9,1)
        return [weights0,weights1,weights2,weights3,weights4,weights5,weights6,weights7,weights8,weights9]

    def get_free_parts(self,normalize=False):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g0 = Function('g')
        g0 = -2*exp(1.5*x1)*u0
        if normalize:
            g0 = g0 / 2
        return [g0]*10

    def get_expression(self):
        return super().get_expression()
    

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        death_rate = lambda x: 2*np.exp(1.5*x)
        birth_rate = lambda x: np.where(x < 1,np.sin(x*np.pi),np.zeros_like(x))
        initial_age_distribution = boundary_functions[0]

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

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = -exp(C*X1)*X2

        return [g]*10


class BurgerDict(PDE):
    
    def __init__(self,v=0.1):
        super().__init__(None)
        self.v = v
    
    @property
    def name(self):
        return "Burger"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1

    @property
    def num_conditions(self):
        return 1

    def get_dictionaries(self):
        dict0 = [
            ED(Partial([0,0]),proj_0),
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),square_0),
            ED(Partial([0,2]),proj_0)
        ]
        return [dict0]
    def get_weights(self,normalize=False):
        weights0 = np.array([0.0,1.0,0.0,0.5,0.0,-self.v])
        # weights0 = np.array([1.0,0.5,-self.v])
        if normalize:
            weights0 /= np.linalg.norm(weights0,1)
        return [weights0]

    def get_sindy_weights(self):
        return [np.array([1.0,0.0,0.0,self.v,-1.0,0.0])]

    def get_free_parts(self,normalize=False):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g0 = Function('g')
        g0 = 0 * x1
        return [g0]

    def get_expression(self):
        return super().get_expression()
    

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        initial_cond = boundary_functions[0]

        def func(grid):
            assert grid.num_dims == 2
            axes = grid.axes
            widths = grid.widths
            delta_t = 0.002
            delta_x = 0.002
            burger = Burger(self.v,initial_cond)
            U = burger.solve(delta_t,delta_x,widths[0],widths[1])
            sol = np.zeros(grid.shape)
            grid_trans = grid.as_grid()
            for t, g_t in enumerate(grid_trans):
                for a, g_t_a in enumerate(g_t):
                    ind_t = int(g_t_a[0] / delta_t)
                    ind_a = int(g_t_a[1] / delta_x)
                    sol[t,a] = U[ind_t, ind_a]

            return sol

        return [func]

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = C

        return [g]


class KSDict(PDE):
    
    def __init__(self,v=0.1):
        super().__init__(None)
        self.v = v
    
    @property
    def name(self):
        return "Burger"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1

    @property
    def num_conditions(self):
        return 1

    def get_dictionaries(self):
        dict0 = [
            ED(Partial([0,0]),proj_0),
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([0,2]),square_0),
            ED(Partial([0,3]),proj_0),
            ED(Partial([0,3]),square_0),
            ED(Partial([0,4]),proj_0),
            ED(Partial([0,4]),square_0)
        ]
        return [dict0]
    def get_weights(self,normalize=False):
        weights0 = np.array([0.0,1.0,0.0,0.5,1.0,0.0,0.0,0.0,1.0,0.0])
        # weights0 = np.array([1.0,0.5,-self.v])
        if normalize:
            weights0 /= np.linalg.norm(weights0,1)
        return [weights0]

    def get_sindy_weights(self):
        return [np.array([1.0,0.0,0.0,-1.0,0,-1,-1,0,0,0])]

    def get_free_parts(self,normalize=False):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g0 = Function('g')
        g0 = 0 * x1
        return [g0]

    def get_expression(self):
        return super().get_expression()
    

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        initial_cond = boundary_functions[0]
        import scipy.io
        data = scipy.io.loadmat('experiments/data/KS/kuramoto_sivishinky.mat')
        u = data['uu']
        x = data['x'][:,0]
        t = data['tt'][0,:]
        dt = t[1]-t[0]
        dx = x[2]-x[1]

        def func(grid):
            assert grid.num_dims == 2
            axes = grid.axes
            widths = grid.widths
            delta_t = dt
            delta_x = dx

            sol = np.zeros(grid.shape)
            grid_trans = grid.as_grid()
            for t, g_t in enumerate(grid_trans):
                for a, g_t_a in enumerate(g_t):
                    # print(g_t_a[0])
                    # print(delta_t)
                    # print(g_t_a[1])
                    # print(delta_x)
                    ind_t = int(g_t_a[0] / delta_t)
                    ind_a = int(g_t_a[1] / delta_x)
                    sol[t,a] = u[ind_a, ind_t]

            return sol

        return [func]

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = C

        return [g]


class NS(PDE):
    
    def __init__(self):
        super().__init__(None)
    
    @property
    def name(self):
        return "NS"

    @property
    def M(self):
        return 3

    @property
    def N(self):
        return  3

    @property
    def num_conditions(self):
        return 1

    def get_dictionaries(self):

        proj_2 = EFunction(lambda vx,vu: vu[2], "u_2")
        prod_0_1 = EFunction(lambda vx,vu: vu[0]*vu[1], "(u_0)*(u_1)")
        prod_0_2 = EFunction(lambda vx,vu: vu[0]*vu[2], "(u_0)*(u_2)")
        prod_1_2 = EFunction(lambda vx,vu: vu[1]*vu[2], "(u_1)*(u_2)")

        dict0 = [
           ED(Partial([1,0,0]),proj_2),
           ED(Partial([0,1,0]),prod_0_2),
           ED(Partial([0,0,1]),prod_1_2),
           ED(Partial([0,2,0]),proj_2),
           ED(Partial([0,0,2]),proj_2)
        ]
        dict1 = [
            ED(Partial([0,1,0]),proj_0),
            ED(Partial([0,0,1]),proj_1)
        ]
        dict2 = [
            ED(Partial([0,0,0]),proj_2),
            ED(Partial([0,1,0]),proj_1),
            ED(Partial([0,0,1]),proj_0)
        ]
        return [dict0,dict1,dict2]
    def get_weights(self,normalize=False):
        weights0 = np.array([1.0,1.0,1.0,-0.005,-0.005])
        weights1 = np.array([1.0,1.0])
        weights2 = np.array([1.0,-1.0,1.0])
        
        # weights0 = np.array([1.0,0.5,-self.v])
        if normalize:
            weights0 /= np.linalg.norm(weights0,1)
            weights1 /= np.linalg.norm(weights1,1)
            weights2 /= np.linalg.norm(weights2,1)
        return [weights0, weights1,weights2]

    def get_sindy_weights(self):
        return [np.array([1.0,1.0,1.0,-0.005,-0.005])]

    def get_free_parts(self,normalize=False):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g0 = Function('g')
        g0 = 0 * x1
        return [g0]*3

    def get_expression(self):
        return super().get_expression()
    

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        initial_cond = boundary_functions[0]

        with open('ns_data.p','rb') as file:
            data = pickle.load(file)

        u = data['u']
        dt = data['dt']
        dx = data['dx']
        dy = data['dy']
        nx = data['nx']
        

        def func_with_axis(return_axis):

            def func(grid):
                assert grid.num_dims == 3
                axes = grid.axes
                widths = grid.widths
                delta_t = dt
                delta_y = dy
                delta_x = dx
                

                sol_u = np.zeros(grid.shape)
                sol_v = np.zeros(grid.shape)
                sol_rot = np.zeros(grid.shape)
                grid_trans = grid.as_grid()
                for t, g_t in enumerate(grid_trans):
                    for x, g_t_x in enumerate(g_t):
                        for y, g_t_x_y in enumerate(g_t_x):
                            ind_t = min(int(g_t_x_y[0] / delta_t),nx-1)
                            ind_x = min(int(g_t_x_y[1] / delta_x),nx-1)
                            ind_y = min(int(g_t_x_y[2] / delta_y),nx-1)

                            sol_u[t,x,y] = u[0,ind_t, ind_x, ind_y]
                            sol_v[t,x,y] = u[1,ind_t, ind_x, ind_y]
                            sol_rot[t,x,y] = u[2,ind_t, ind_x, ind_y]
                
                if return_axis == 0:
                    return sol_u
                elif return_axis == 1:
                    return sol_v
                elif return_axis == 2:
                    return sol_rot
            
            return func

        return [func_with_axis(0),func_with_axis(1),func_with_axis(2)]

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = C

        return [g]*3


class KdVDict(PDE):
    
    def __init__(self):
        super().__init__(None)
    
    @property
    def name(self):
        return "KdV"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1

    @property
    def num_conditions(self):
        return 2

    def get_dictionaries(self):
        dict1 = [
            ED(Partial([0,0]),proj_0),
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),square_0),
            ED(Partial([0,2]),proj_0),
            ED(Partial([0,3]),proj_0),
            ED(Partial([0,3]),square_0)
        ]
        dict0 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,3]),proj_0)
        ]
        return [dict0]
    def get_weights(self,normalize=False):
        weights1 = np.array([0,1.0,0.0,3.0,0,0,1.0,0])
        weights0 = np.array([1.0,3.0,1.0])
        if normalize:
            weights0 /= np.linalg.norm(weights0,1)
        return [weights0]

    def get_free_parts(self,normalize=False):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g0 = Function('g')
        g0 = 0 * x1
        return [g0]

    def get_expression(self):
        return super().get_expression()
    

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        c = boundary_functions[0]
        a = boundary_functions[1]

        def func(grid):
            assert grid.num_dims == 2
            x = grid.by_axis()
            sol = c/2*np.cosh(np.sqrt(c)/2*(x[1]-c*x[0]-a))**-2

            return sol

        return [func]

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = C

        return [g]

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

class Liouville_L1(PDE):

    def __init__(self, d):
        super().__init__({'d': d})

    @property
    def name(self):
        return "Liouville_L1"

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
        L = LinearOperator([1.0,1.0],[Partial([2,0]),Partial([0,2])])
        g = exp(self.params['d']*u0)
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        a = 1.2
        b = boundary_functions[1]
        # c = 0.55
        c = 10.0
        d = self.params['d']
        def func(grid):
            x = grid.by_axis()[0]
            y = grid.by_axis()[1]
            # return np.log((2*(a*np.sin(x)*np.sinh(y) + b*np.cos(x)*np.sinh(y))**2 \
            # + 2*(a*np.cos(x)*np.cosh(y) - b*np.sin(x)*np.cosh(y))**2)/(d*(a*np.sin(x)*np.cosh(y) \
            # + b*np.cos(x)*np.cosh(y) + c)**2))/d
            return np.log((2*a**2*(-2*y - 2)**2 + 2*(a*(2*x + 2) + b)**2)/(d*(a*((x + 1)**2 - (y + 1)**2) + b*(x + 1) + c)**2))/d

        return [func]

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
        
        g = exp(C*X2)

        return [g]

class Liouville2_L1(PDE):

    def __init__(self, d):
        super().__init__({'d': d})

    @property
    def name(self):
        return "Liouville2_L1"

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
        u0 = symbols('u0', real=True)
        g = Function('g')
        L = LinearOperator([0.5,0.5],[Partial([2,0]),Partial([0,2])])
        g = exp(self.params['d']*u0)
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
        
        def func(grid):
            x = grid.by_axis()[0]
            y = grid.by_axis()[1]
            return boundary_functions[0](x, y)
        return [func]

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
        
        g = exp(C*X2)

        return [g]    


class HeatEquationHomo(PDE):

    def __init__(self, k):
        super().__init__({'k': k})

    @property
    def name(self):
        return "HeatEquationHomo"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  1

    @property
    def num_conditions(self):
        return 1

    def get_dictionaries(self):
        dict0 = [
            ED(Partial([0,0]),proj_0),
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([0,1]),square_0),
            ED(Partial([0,2]),square_0),
            ED(Partial([0,2]),proj_0)
        ]
        return [dict0]
    def get_weights(self,normalize=False):
        weights0 = np.array([0.0,1.0,0.0,0.0,0.0,-self.params['k']])
        # weights0 = np.array([1.0,0.5,-self.v])
        if normalize:
            weights0 /= np.linalg.norm(weights0,1)
        return [weights0]

    def get_sindy_weights(self):
        return [np.array([1.0,0.0,0.0,self.params['k'],0.0,0.0])]

    def get_free_parts(self,normalize=False):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g0 = Function('g')
        g0 = 0 * x1
        return [g0]

    def get_expression(self):
        # x0,x1 = symbols('x0,x1', real=True)
        # g = Function('g')
        # L = LinearOperator([1.0,-self.params['k']],[Partial([1,0]),Partial([0,2])])
        # g = (1+np.abs(self.params['k'])) * exp(self.params['theta']*x0)
        # return [(L,g)]
        pass
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        heat_source = lambda X: np.zeros_like(X[0])
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
      
        g = C

        return [g]

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

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = 0*X0

        return [g]

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
        L = LinearOperator([1.0,-self.params['k']**2,self.params['d']],[Partial([2,0]),Partial([0,2]),Partial([1,0])])
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

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = 0*X0

        return [g]


class WaveEquation3_L1(PDE):

    def __init__(self, k):
        super().__init__({'k': k})

    @property
    def name(self):
        return "WaveEquation3_L1"

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
        g = 2*exp(x0)*sin(3*x0)
        return [(L,g)]
    
    def get_solution(self, boundary_functions):

        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")

        wave_source = lambda X: 2*np.exp(X[0])*np.sin(3*X[0])
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

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = exp(X0)*sin(C*X0)

        return [g]


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

class FullFlow2D(PDE):

    def __init__(self):
        super().__init__(None)
    
    @property
    def name(self):
        return "FullFlow2D"

    @property
    def M(self):
        return 2

    @property
    def N(self):
        return  2

    @property
    def num_conditions(self):
        return 2

    def get_dictionaries(self):
        
        dict0 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_0),
            ED(Partial([2,0]),proj_0),
            ED(Partial([0,2]),proj_0)
        ]
        dict1 = [
            ED(Partial([1,0]),proj_1),
            ED(Partial([0,1]),proj_1),
            ED(Partial([2,0]),proj_1),
            ED(Partial([0,2]),proj_1)
        ]
        dict2 = [
            ED(Partial([1,0]),proj_0),
            ED(Partial([0,1]),proj_1),
            ED(Partial([2,0]),proj_0),
            ED(Partial([0,2]),proj_1)
        ]
        dict3 = [
            ED(Partial([1,0]),proj_1),
            ED(Partial([0,1]),proj_0),
            ED(Partial([2,0]),proj_1),
            ED(Partial([0,2]),proj_0)
        ]
        return [dict0,dict1,dict2,dict3]
    def get_weights(self,normalize=False):
        weights0 = np.array([0.0,0.0,1.0,1.0])
        weights1 = np.array([0.0,0.0,1.0,1.0])
        weights2 = np.array([1.0,-1.0,0.0,0.0])
        weights3 = np.array([1.0,1.0,0.0,0.0])
        if normalize:
            weights0 /= np.linalg.norm(weights0,1)
            weights1 /= np.linalg.norm(weights1,1)
            weights2 /= np.linalg.norm(weights2,1)
            weights3 /= np.linalg.norm(weights3,1)
        return [weights0,weights1,weights2,weights3]

    def get_free_parts(self,normalize=False):
        x0,x1 = symbols('x0,x1', real=True)
        u0 = symbols('u0', real=True)
        g0 = Function('g')
        g0 = 0 * x1
        return [g0,g0,g0,g0]

    
    def get_expression(self):
        x0,x1 = symbols('x0,x1', real=True)
        g = Function('g')
        L = LinearOperator([1.0,1.0],[Partial([2,0]),Partial([0,2])])
        g = 0.0*x0 + 0.0*x1
        return [(L,g)]

    def get_solution(self, boundary_functions):
        if len(boundary_functions) != self.num_conditions:
            raise ValueError("Wrong number of boundary functions")
    
        def func_potential(grid):
            locs = boundary_functions[0]
            strengths = boundary_functions[1]
            locs[:,0] *= grid.widths[0]
            locs[:,1] *= grid.widths[1]
            return get_flow_potential_2D(grid,locs,strengths)
            # return get_complex_potential_re(grid,locs,strengths)
        
        def func_stream(grid):
            locs = boundary_functions[0]
            strengths = boundary_functions[1]
            locs[:,0] *= grid.widths[0]
            locs[:,1] *= grid.widths[1]
            return get_flow_stream_2D(grid,locs,strengths)
            # return get_complex_potential_im(grid,locs,strengths)


        return [func_potential, func_stream]

    def get_functional_form_normalized(self,norm='l1'):
        X0 = Symbol('X0', real=True)
        X1 = Symbol('X1', real=True)
        X2 = Symbol('X2', real=True)
        C = Symbol('C', real=True, positive=True)
      
        g = C

        return [g,g,g,g]

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


