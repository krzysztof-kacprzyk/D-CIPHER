from matplotlib import animation, pyplot as plt
import numpy as np

from var_objective.conditions import get_conditions_set

from .libs import tdma

from matplotlib import pyplot as plt
from matplotlib import animation

class HeatEquationNeumann1D():

    def __init__(self, k, heat_source, boundary1, boundary2, initial_temp):

        self.k = k
        self.heat_source = heat_source
        self.boundary1 = boundary1
        self.boundary2 = boundary2
        self.initial_temp = initial_temp

    def btcs(self, max_t, max_x, delta_t, delta_x):

        M = int(max_t / delta_t) + 1
        N = int(max_x / delta_x) + 1

        U = np.zeros((M,N))

        r = self.k * delta_t / (delta_x ** 2)
        
        time_axis = np.linspace(0,max_t,M)
        x_axis = np.linspace(0,max_x,N)

        init = self.initial_temp(x_axis)
        b1 = self.boundary1(time_axis)
        b2 = self.boundary2(time_axis)

        mgrid = np.meshgrid(time_axis,x_axis,indexing='ij')
        source = self.heat_source(mgrid)

        U[0,:] = init

        for i in range(1,M):

            W = np.zeros((N,N))
            W[0,0] = 1 + 2*r
            W[0,1] = -2*r

            for j in range(1,N-1):
                W[j,j-1] = -r
                W[j,j] = 1+2*r
                W[j,j+1] = -r
            
            W[N-1,N-2] = -2*r
            W[N-1,N-1] = 1+2*r

            boundary = np.zeros(N)
            boundary[0] = 2*r*delta_x*b1[i]
            boundary[N-1] = -2*r*delta_x*b2[i]

            b = boundary - delta_t*source[i,:]

            u = tdma(W, U[i-1,:]-b)

            U[i,:] = u

        return U

if __name__ == "__main__":

    conditions = get_conditions_set('Heat1')
    widths = [2.0,2.0]

    for condition in conditions.conditions:


        k = 0.1
        # heat_source = lambda X: 4*np.sin(2*np.pi*X[1])
        heat_source = lambda X: (X[1] - 1.0) ** 2 * (1.0 - X[0])
        boundary1  = lambda x: np.zeros_like(x)
        boundary2 = lambda x: np.zeros_like(x)
        # initial_temp = lambda x: 8 * (x-0.5) * (x-0.5)
        initial_temp = condition[0]

        heat_eq = HeatEquationNeumann1D(k,heat_source,boundary1,boundary2,initial_temp)

        sol = heat_eq.btcs(widths[0],widths[1],0.001,0.001)

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
