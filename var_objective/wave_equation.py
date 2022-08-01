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
        self.mgrid = mgrid
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
        self.mgrid = mgrid
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


    
    # conditions = get_conditions_set('HeatRandom',params={'num_samples':10,'seed':437782})
    # widths = [2.0,2.0]

    # chosen_c = 6

    # settings = []

    # org_setting = {'name':'org','k':1.0,'source':lambda X: 2*np.exp(X[0])*np.sin(3*X[0])}

    # settings.append({'name':'var_0.001','k':1.0,'source':lambda X: 2.021*np.exp(X[0])*np.sin(2.9772*X[0])})

    # settings.append({'name':'var_0.01','k':1.0011,'source':lambda X: 2.1124*np.exp(0.9711*X[0])*np.sin(2.9706*X[0])})

    # settings.append({'name':'mse_0.001','k':0.9879,'source':lambda X: 2.5227*(-X[0]+2*np.sin(np.exp(X[0])))})

    # settings.append({'name':'mse_0.01','k':0.9839,'source':lambda X: 2.9860*(-X[0]+np.sin(np.exp(X[0])))})

    # settings.append({'name':'var_0.015','k':1.0,'source':lambda X: 2.1763*np.exp(0.9504*X[0])*np.sin(2.9439*X[0])})

    # settings.append({'name':'mse_0.015','k':0.9779,'source':lambda X: 3.1636*np.log(np.abs(np.log(0.7137*X[0])))})

    # # settings.append({'name':'var_0.05','k':0.9914,'source':lambda X: -2.9700*(X[0]*(X[0]-0.1066))})
    
    # vmax = 3.0
    # vmin = 0
    # for c in range(conditions.get_num_samples()):

    #     if c != chosen_c:
    #         continue

    #     condition = conditions.get_condition_functions(c)

        
    #     # damping_coeff = 1.0
        
    #     # f = k/(2*2.0)
    #     # wave_source = lambda X: (X[1] - 1.0) ** 2 * (1.0 - X[0])
    #     # wave_source = lambda X: 100*np.sin(10*X[0]) * np.where(np.logical_and(0.9 < X[1],X[1] < 1.1),1.0,0.0)
    #     # wave_source = lambda X: 10*np.sin(2*np.pi*f*X[0])*np.sin(np.pi*X[1]/2.0)
    #     # wave_source = lambda X: np.zeros_like(X[1])
        
    #     # initial_temp = lambda x: 8 * (x-0.5) * (x-0.5)
    #     org_k = org_setting['k']
    #     org_source = org_setting['source']

    #     initial_wave = condition[0]

    #     # wave_eq = DampedWaveEquationDirichlet1D(k,damping_coeff,wave_source,initial_wave)
    #     org_wave_eq = WaveEquationDirichlet1D(org_k,org_source,initial_wave)

    #     org_sol = org_wave_eq.idm(widths[0],widths[1],0.001,0.001)

    #     # a = sol.shape[0]
    #     # b = sol.shape[1]
    #     # print(sol.shape)

    #     # y_max = sol.max()
    #     # y_min = sol.min()

    #     # mgrid = wave_eq.mgrid
        
    #     # plt.contourf(mgrid[0],mgrid[1],sol,levels=30)

    #     for setting in settings:
    #         k = setting['k']
    #         source = setting['source']
    #         name = setting['name']
            
    #         wave_eq = WaveEquationDirichlet1D(k,source,initial_wave)

    #         sol = wave_eq.idm(widths[0],widths[1],0.001,0.001)

    #         fig = plt.figure()

    #         plt.imshow(np.transpose(np.abs(sol-org_sol)),cmap=plt.get_cmap('gist_heat_r'),interpolation='nearest',vmin=vmin,vmax=vmax)
    #         plt.colorbar()

    #         plt.savefig(f'results/figs/{c}/{name}_{c}_diff.png')


        # # First set up the figure, the axis, and the plot element we want to animate
        # fig = plt.figure()
        # ax = plt.axes(xlim=(0, widths[1]), ylim=(y_min,y_max))
        # line, = ax.plot([], [], lw=2)

        # # initialization function: plot the background of each frame
        # def init():
        #     line.set_data([], [])
        #     return line,

        # # animation function.  This is called sequentially
        # def animate(i):
            
        #     line.set_data(np.linspace(0.0,widths[1],b), sol[i,:])
        #     return line,

        # # call the animator.  blit=True means only re-draw the parts that have changed.
        # anim = animation.FuncAnimation(fig, animate, init_func=init,
        #                             frames=a, interval=1, blit=True)

       
        # plt.show()

def create_plot():

    conditions = get_conditions_set('HeatRandom',params={'num_samples':10,'seed':437782})
    widths = [2.0,2.0]

    chosen_c = 6

    settings = [[],[]]

    org_setting = {'name':'org','k':1.0,'source':lambda X: 2*np.exp(X[0])*np.sin(3*X[0])}

    settings[0].append({'name':'var_0.001','k':1.0,'source':lambda X: 2.021*np.exp(X[0])*np.sin(2.9772*X[0])})

    settings[0].append({'name':'var_0.01','k':1.0011,'source':lambda X: 2.1124*np.exp(0.9711*X[0])*np.sin(2.9706*X[0])})

    settings[1].append({'name':'mse_0.001','k':0.9879,'source':lambda X: 2.5227*(-X[0]+2*np.sin(np.exp(X[0])))})

    settings[1].append({'name':'mse_0.01','k':0.9839,'source':lambda X: 2.9860*(-X[0]+np.sin(np.exp(X[0])))})

    settings[0].append({'name':'var_0.015','k':1.0,'source':lambda X: 2.1763*np.exp(0.9504*X[0])*np.sin(2.9439*X[0])})

    settings[1].append({'name':'mse_0.015','k':0.9779,'source':lambda X: 3.1636*np.log(np.abs(np.log(0.7137*X[0])))})

    condition = conditions.get_condition_functions(chosen_c)

    org_k = org_setting['k']
    org_source = org_setting['source']

    initial_wave = condition[0]

    org_wave_eq = WaveEquationDirichlet1D(org_k,org_source,initial_wave)

    org_sol = org_wave_eq.idm(widths[0],widths[1],0.001,0.001)


    fig, axs = plt.subplots(2, 3,figsize=(14,7),dpi=150)
    vmin = 0
    vmax = 3.0
    subtitle_fontsize = 14
    noise_fontsize = 15
    algo_fontsize = 15
    label_fontsize = 13
    barticks_fontsize = 14
    eqs = [[r'$2.02\times e^{t}\sin(2.98t)$',r'$2.11\times e^{0.97t}\sin(2.97t)$',r'$2.17\times e^{0.95t}\sin(2.94t)$'],
        [r'$2.52\times (2\times \sin(e^{t})-t)$',r'$2.99\times (\sin(e^{t})-t)$',r'$3.16 \times \log(|\log(0.71 \times t)|)$']]
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel(r'$t$',fontsize=label_fontsize)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel(r'$x$',fontsize=label_fontsize)
            ax.title.set_text(eqs[i][j])
            ax.title.set_fontsize(subtitle_fontsize)

            k = settings[i][j]['k']
            source = settings[i][j]['source']
            name = settings[i][j]['name']
            
            wave_eq = WaveEquationDirichlet1D(k,source,initial_wave)
            sol = wave_eq.idm(widths[0],widths[1],0.001,0.001)

            im = ax.imshow(np.transpose(np.abs(sol-org_sol)),cmap=plt.get_cmap('gist_heat_r'),interpolation='nearest',vmin=vmin,vmax=vmax)
            
    fig.subplots_adjust(right=0.65)
    cbar_ax = fig.add_axes([0.70, 0.15, 0.04, 0.7])
    b = fig.colorbar(im,cax=cbar_ax)
    b.set_ticks([0,3.0])
    b.set_ticklabels(['0','max'])
    b.ax.tick_params(labelsize=barticks_fontsize)
    fig.text(0, 0.69, "D-CIPHER", fontsize=algo_fontsize)
    fig.text(0, 0.28, "Abl. D-CIPHER", fontsize=algo_fontsize)
    fig.text(0.17, 0.94, r'$\sigma_R=10^{-3}$',fontsize=noise_fontsize)
    fig.text(0.35, 0.94, r'$\sigma_R=10^{-2}$',fontsize=noise_fontsize)
    fig.text(0.515, 0.94, r'$\sigma_R=1.5\times10^{-2}$',fontsize=noise_fontsize)

    plt.savefig(f'results/figs/comparison.pdf')



if __name__ == "__main__":

    create_plot()