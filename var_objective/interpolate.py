from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from scipy.interpolate import UnivariateSpline, SmoothBivariateSpline

np.random.seed(0)


def estimate_fields(observed_grid, observed_dataset, full_grid, seed=0, method='gp'):
    
    D = observed_dataset.shape[0] # number of samples
    N = observed_dataset.shape[1] # number of dimensions

    # Configure Gaussian Process
    kernel = RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=seed)


    

    samples_list = []
    for d in range(D):

        U_est = np.zeros([N,*full_grid.shape])
    
        for j in range(N):

            u_obs_j = observed_dataset[d][j].flatten()
            mean = np.mean(u_obs_j)
            std = np.std(u_obs_j)
            if method == 'gp':
                gpr.fit(observed_grid.as_covariates(), (u_obs_j - mean)/std)
                u_pred_j = gpr.predict(full_grid.as_covariates()) * std + mean
            elif method == 'spline':
                if len(observed_grid.shape) == 1:
                    spline = UnivariateSpline(observed_grid.as_covariates().flatten(),(u_obs_j - mean)/std,bbox=[0,observed_grid.widths[0]],s=0.01)
                    u_pred_j = spline(full_grid.as_covariates().flatten()) * std + mean
                elif len(observed_grid.shape) == 2:
                    spline = SmoothBivariateSpline(observed_grid.as_covariates()[:,0],observed_grid.as_covariates()[:,1],(u_obs_j - mean)/std,bbox=[0,observed_grid.widths[0],0,observed_grid.widths[1]],s=0.01)
                    u_pred_j = spline(full_grid.as_covariates()[:,0],full_grid.as_covariates()[:,1], grid=False) * std + mean
                else:
                    raise ValueError("Spline interpolation is implemented only up to 2 dimensions")
            elif method == 'none':
                u_pred_j = u_obs_j
            else:
                raise ValueError(f"Interpolation method {method} not implmented")
            u_pred_j = full_grid.from_labels_to_grid(u_pred_j)
            U_est[j] = u_pred_j

            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # ax.scatter(observed_grid.by_axis()[0], observed_grid.by_axis()[1],u_obs_j, cmap=cm.coolwarm)
            # plt.show()

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # Plot the surface.
            surf = ax.plot_surface(full_grid.by_axis()[0], full_grid.by_axis()[1], u_pred_j, linewidth=0, antialiased=False, cmap=cm.coolwarm)
            ax.scatter(observed_grid.by_axis()[0], observed_grid.by_axis()[1],u_obs_j)
            plt.show()


           
            # plt.plot(full_grid.by_axis()[0], u_pred_j)
            # plt.scatter(observed_grid.by_axis()[0],u_obs_j)
            # plt.show()

        samples_list.append(U_est)

    return np.stack(samples_list, axis=0)

if __name__ == "__main__":

    from .grids import EquiPartGrid
 

    g = EquiPartGrid([5.0], 1000)

    obs = EquiPartGrid([5.0],10)

    X_obs = obs.by_axis()
    print(X_obs.shape)
    V_obs =  np.expand_dims(np.concatenate([np.sin(X_obs) + np.random.normal(0,0.2,size=X_obs.shape), 3+np.cos(X_obs) + np.random.normal(0,0.2,size=X_obs.shape)], axis=0),axis=0)
    original = 3+np.cos(g.as_grid().flatten())

    # X_obs = np.array([[0.2],[0.5],[0.7],[3.1]])
    # U_obs = np.array([[0.6],[1.2],[1.6],[2.0]])

    U_est = estimate_fields(obs, V_obs, g)

    # print(g.as_grid())
    # print(U_pred)



    plt.scatter(X_obs.flatten(), V_obs[0,1,:].flatten())
    plt.plot(g.as_grid().flatten(), U_est[0,1,:].flatten(), label="GP")
    plt.plot(g.as_grid().flatten(), original, label='orig')
    plt.legend()
    plt.show()