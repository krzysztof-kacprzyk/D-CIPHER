from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import numpy as np

np.random.seed(0)


def estimate_fields(observed_grid, observed_dataset, integration_grid, seed=0):
    
    D = observed_dataset.shape[0] # number of samples
    N = observed_dataset.shape[1] # number of dimensions

    # Configure Gaussian Process
    kernel = RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=seed)


    

    samples_list = []
    for d in range(D):

        U_est = np.zeros([N,*integration_grid.shape])
    
        for j in range(N):

            u_obs_j = observed_dataset[d][j].flatten()
            gpr.fit(observed_grid.as_covariates(), u_obs_j)
            u_pred_j = gpr.predict(integration_grid.as_covariates())
            u_pred_j = integration_grid.from_labels_to_grid(u_pred_j)
            U_est[j] = u_pred_j

        samples_list.append(U_est)

    return np.stack(samples_list, axis=0)


# from .grids import EquiPartGrid
# import matplotlib.pyplot as plt

# g = EquiPartGrid([5.0], 1000)

# X_obs = np.expand_dims(np.linspace(0,5,30),axis=-1)
# V_obs =  np.concatenate([np.sin(X_obs) + np.random.normal(0,0.4,size=X_obs.shape), np.cos(X_obs) + np.random.normal(0,0.4,size=X_obs.shape)], axis=1)
# original = np.cos(g.as_grid().flatten())

# # X_obs = np.array([[0.2],[0.5],[0.7],[3.1]])
# # U_obs = np.array([[0.6],[1.2],[1.6],[2.0]])

# U_est = estimate_fields(X_obs, V_obs, g)

# # print(g.as_grid())
# # print(U_pred)



# plt.scatter(X_obs.flatten(), V_obs[:,1].flatten())
# plt.plot(g.as_grid().flatten(), U_est[1].flatten(), label="GP")
# plt.plot(g.as_grid().flatten(), original, label='orig')
# plt.legend()
# plt.show()