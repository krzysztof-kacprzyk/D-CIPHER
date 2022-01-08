import numpy as np

class SLM:
    """
        Sharpe-Lotka-McKendrick model
    """
    def __init__(self, death_rate, birth_rate, initial_age_distribution):
        self.death_rate = death_rate
        self.birth_rate = birth_rate
        self.initial_age_distribution = initial_age_distribution

    def solve_second_order(self, max_time, delta_t, initial_max_age):
        N = int(max_time / delta_t)
        L = int(initial_max_age / delta_t) + 1
        U = np.zeros((N+1,L+N+1))

        beta = self.birth_rate(np.linspace(0, (L+N)*delta_t,L+N+1))
        mu = self.death_rate(np.linspace(delta_t/2, (L+N)*delta_t - delta_t/2, L+N))

        for i in range(L+1):
            U[0,i] = self.initial_age_distribution(i * delta_t)

        for n in range(1, N+1):

            U[n,1:] = U[n-1,:-1] * ((2-mu*delta_t)/(2+mu*delta_t))
            U[n,0] = np.sum(U[n,1:]*beta[1:]*delta_t) + (delta_t / 2) * beta[0] * U[n-1,0]

        return U



if __name__ == "__main__":

    # d = lambda x: 2.0*np.exp(1.0*x)
    # b = lambda x: 0.0025*x*np.exp(-0.05*x)

    # b = lambda x: np.where((x < 0.2) | (x > 0.4), 0, 10*np.exp(-100*np.power(x-0.3,2)))
    
    # b = lambda x: np.where((0.5 < x) & (x < 0.6), 2.0, 0.0)
    # init = lambda x: np.where((x < 0.1), np.exp(-1000*np.power(x-0.05,2)),0)


    d = lambda x: 2*np.exp((x-1))
    b = lambda x: 5*(-np.power(2*x-1,2)+1)
    init = lambda x: np.power(x-1,2)

    slm = SLM(d,b,init)

    sol = slm.solve_second_order(1.0,0.001,1.0)

    a = sol.shape[0]
    b = sol.shape[1]


    from matplotlib import pyplot as plt
    from matplotlib import animation

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1.0), ylim=(0.0,10.0))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        
        line.set_data(np.linspace(0.0,2.0,b), sol[i,:])
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=a, interval=1, blit=True)

    plt.show()