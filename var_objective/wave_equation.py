from matplotlib import animation, pyplot as plt
import numpy as np

from var_objective.conditions import get_conditions_set

from .libs import tdma

from matplotlib import pyplot as plt
from matplotlib import animation

class WaveEquationDirichlet1D():

    """
        u_tt = k^2 * u_xx + s(t,x)
        u(0,x) = initial_wave(x)
        u(t,0) = initial_wave(0)
        u(t,end) = initial_wave(end)
    """

    def __init__(self, k, wave_source, initial_wave):

        self.k = k
        self.wave_source = wave_source
        self.initial_wave = initial_wave

    def idm(self, max_t, max_x, delta_t, delta_x):

        M = int(max_t / delta_t) + 1
        N = int(max_x / delta_x) + 1

        U = np.zeros((M,N))

        r = self.k * delta_t / delta_x
        
        time_axis = np.linspace(0,max_t,M)
        x_axis = np.linspace(0,max_x,N)

        init = self.initial_wave(x_axis)

        mgrid = np.meshgrid(time_axis,x_axis,indexing='ij')
        source = self.wave_source(mgrid)

        b_start = init[0]
        b_end = init[-1]

        U[0,:] = init

        # Apply boundary conditions
        U[:,0] = b_start
        U[:,-1] = b_end

        # Solve only for x in [1,N-2] as 0 and N-1 is the boundary
        N_new = N-2
        W = np.zeros((N_new,N_new))
        W[0,0] = 2
        W[0,1] = -1

        for j in range(1,N_new-1):
            W[j,j-1] = -1
            W[j,j] = 2
            W[j,j+1] = -1
        
        W[N_new-1,N_new-2] = -1
        W[N_new-1,N_new-1] = 2

        M1 = 4/(r ** 2) * np.eye(N_new) + W
        M2 = 2*(4/(r ** 2) * np.eye(N_new) - W)
        M3 = 4/(r ** 2) * np.eye(N_new) + W
        M4 = 4/(r ** 2) * np.eye(N_new) - 3*W


        b = np.zeros(N_new)
        b[0] = -b_start
        b[N_new-1] = -b_end

        # A different equation for t = 1. We use u_t(0,x) = 0
        source_term = 4 * source[1,1:-1]*(delta_x ** 2) / (self.k ** 2)
        u_1 = tdma(M1,np.dot(M4,U[0,1:-1]) - 4*b + source_term)
        U[1,1:-1] = u_1

        for i in range(2,M):
            source_term = 4 * source[i,1:-1]*(delta_x ** 2) / (self.k ** 2)
            u = tdma(M1, np.dot(M2,U[i-1,1:-1])-np.dot(M3,U[i-2,1:-1])-4*b+source_term)

            U[i,1:-1] = u

        return U


class DampedWaveEquationDirichlet1D():

    """
        u_tt = k^2 * u_xx + s(t,x)
        u(0,x) = initial_wave(x)
        u(t,0) = initial_wave(0)
        u(t,end) = initial_wave(end)
    """

    def __init__(self, k, damping_coeff, wave_source, initial_wave):

        self.k = k
        self.wave_source = wave_source
        self.initial_wave = initial_wave
        self.d = damping_coeff

    def idm(self, max_t, max_x, delta_t, delta_x):

        M = int(max_t / delta_t) + 1
        N = int(max_x / delta_x) + 1

        U = np.zeros((M,N))

        r = self.k * delta_t / delta_x
        
        time_axis = np.linspace(0,max_t,M)
        x_axis = np.linspace(0,max_x,N)

        init = self.initial_wave(x_axis)

        mgrid = np.meshgrid(time_axis,x_axis,indexing='ij')
        source = self.wave_source(mgrid)

        b_start = init[0]
        b_end = init[-1]

        U[0,:] = init

        # Apply boundary conditions
        U[:,0] = b_start
        U[:,-1] = b_end

        # Solve only for x in [1,N-2] as 0 and N-1 is the boundary
        N_new = N-2
        W = np.zeros((N_new,N_new))
        W[0,0] = 2
        W[0,1] = -1

        for j in range(1,N_new-1):
            W[j,j-1] = -1
            W[j,j] = 2
            W[j,j+1] = -1
        
        W[N_new-1,N_new-2] = -1
        W[N_new-1,N_new-1] = 2

        M1 = (4+8*self.d*delta_t)/(r ** 2) * np.eye(N_new) + W
        M2 = 2*(4/(r ** 2) * np.eye(N_new) - W)
        M3 = (4-8*self.d*delta_t)/(r ** 2) * np.eye(N_new) + W
        M4 = (4+24*self.d*delta_t)/(r ** 2) * np.eye(N_new) - 3*W



        b = np.zeros(N_new)
        b[0] = -b_start
        b[N_new-1] = -b_end

        # A different equation for t = 1. We use u_t(0,x) = 0
        source_term = 4 * source[1,1:-1]*(delta_x ** 2) / (self.k ** 2)
        u_1 = tdma(M1,np.dot(M4,U[0,1:-1]) - 4*b + source_term)
        U[1,1:-1] = u_1

        for i in range(2,M):
            source_term = 4 * source[i,1:-1]*(delta_x ** 2) / (self.k ** 2)
            u = tdma(M1, np.dot(M2,U[i-1,1:-1])-np.dot(M3,U[i-2,1:-1])-4*b+source_term)

            U[i,1:-1] = u

        return U

if __name__ == "__main__":

    conditions = get_conditions_set('HeatRandom')
    widths = [2.0,2.0]

    for c in range(conditions.get_num_samples()):

        condition = conditions.get_condition_functions(c)

        k = 15.0
        damping_coeff = 1.0
        
        f = k/(2*2.0)
        # wave_source = lambda X: (X[1] - 1.0) ** 2 * (1.0 - X[0])
        # wave_source = lambda X: 100*np.sin(10*X[0]) * np.where(np.logical_and(0.9 < X[1],X[1] < 1.1),1.0,0.0)
        # wave_source = lambda X: 10*np.sin(2*np.pi*f*X[0])*np.sin(np.pi*X[1]/2.0)
        wave_source = lambda X: np.zeros_like(X[1])
        # initial_temp = lambda x: 8 * (x-0.5) * (x-0.5)
        initial_wave = condition[0]

        wave_eq = DampedWaveEquationDirichlet1D(k,damping_coeff,wave_source,initial_wave)

        sol = wave_eq.idm(widths[0],widths[1],0.001,0.001)

        a = sol.shape[0]
        b = sol.shape[1]
        print(sol.shape)

        y_max = sol.max()
        y_min = sol.min()
        

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(0, widths[1]), ylim=(y_min,y_max))
        line, = ax.plot([], [], lw=2)

        # initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return line,

        # animation function.  This is called sequentially
        def animate(i):
            
            line.set_data(np.linspace(0.0,widths[1],b), sol[i,:])
            return line,

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=a, interval=1, blit=True)

        plt.show()
