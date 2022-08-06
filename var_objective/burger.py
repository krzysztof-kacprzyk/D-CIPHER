import numpy as np

from var_objective.conditions import get_conditions_set
from .libs import tdma

import matplotlib.pyplot as plt
from matplotlib import animation

class Burger:

    def __init__(self,v,initial_cond):
        self.v = v
        self.initial_cond = initial_cond
    
    def solve(self,dt,dx,max_t,max_x):

        time_axis = np.arange(0.0,max_t+dt,dt)
        x_axis = np.arange(0.0,max_x+dx,dx)

        M = len(time_axis)
        N = len(x_axis)

        U = np.zeros((M,N))

        s = self.v*dt/(dx ** 2)

        init = self.initial_cond(x_axis)

        b_start = init[0]
        b_end = init[-1]
        
        # Apply initial conditions
        U[0,1:-1] = init[1:-1]

        # Apply boundary conditions
        U[:,0] = b_start
        U[:,-1] = b_end

        a = np.zeros(N)
        b = np.ones(N) + s
        b[0] = 1
        b[-1] = 1
        c = np.zeros(N)
        d = np.zeros(N)
        for i in range(0,M-1):
            a[1:] = -(dt/(4*dx)) * U[i,:-1] - s/2
            c[:-1] = (dt/(4*dx)) * U[i,1:] - s/2
            d[1:-1] = 0.5*s*U[i,:-2] + (1-s)*U[i,1:-1] + 0.5*s*U[i,2:]


            A = np.diag(b) + np.diag(a[1:],-1) + np.diag(c[:-1],1)
            A = A[1:-1,1:-1]

            u = tdma(A,d[1:-1])
            U[i+1,1:-1] = u

        return U

if __name__ == "__main__":

    conditions = get_conditions_set('BurgerRandom',params={'seed':2,'num_samples':10})
    widths = [2.0,2.0]

    for c in range(conditions.get_num_samples()):

        condition = conditions.get_condition_functions(c)

        v = 0.05
        initial_cond = condition[0]

        burger_eq = Burger(v,initial_cond)

        sol = burger_eq.solve(0.002,0.002,widths[0],widths[1])
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




