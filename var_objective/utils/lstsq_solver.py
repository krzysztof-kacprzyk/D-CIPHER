import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, brentq
import cvxpy as cp
import time
# import torch

from sklearn.linear_model import lars_path_gram
from bisect import bisect_left
import mpmath as mp
from itertools import product
import glob
import pandas as pd

from var_objective.utils.mirror_descent import MirrorDescentSimplex

def ridge(A,b,l):
    n = A.shape[1]
    x = np.dot(np.linalg.inv((np.transpose(A) @ A) + l*np.eye(n)) @ np.transpose(A),b)
    return x


class UnitLstsqSVD:

    def __init__(self, A):
        self.A = A

        m, n = self.A.shape
        self.m = m
        self.n = n
        u,s,vh = np.linalg.svd(A, full_matrices=True)

        s_full = np.zeros(n)
        for i,si in enumerate(s):
            s_full[i]=si

        p = s.shape[0]

        self.main_part = np.transpose(u[:,:p] * s) 

        self.s_full2 = s_full ** 2

        self.v_last = vh[-1,:].flatten()

        self.cov_mat = np.transpose(A) @ A

        mp.dps = 75

    def _check_if_zero(vector):
        if np.sum(vector == 0.0) == len(vector):
            return True
        else:
            return False

    def solve_homogeneous(self, only_loss=False, take_mean=True):

        # Take the square of the smallest singular value
        loss = self.s_full2[-1]
        if take_mean:
            loss /= self.m

        if only_loss:
            return (loss,None)
        else:
            x = self.v_last
            return (loss,x)

    def ridge(self,b,l):
        x = np.dot(np.linalg.inv(self.cov_mat + l*np.eye(self.n)) @ np.transpose(self.A),b)
        return x

    def solve(self,b,only_loss=False,take_mean=True):

        if b is None:
            return self.solve_homogeneous(only_loss=only_loss, take_mean=take_mean)
        
        if UnitLstsqSVD._check_if_zero(b):
            return self.solve_homogeneous(only_loss=only_loss, take_mean=take_mean)

        z = np.dot(self.main_part,b)
        z2 = z ** 2

        z_mod = np.abs(z)

        z2_mask = z2 != 0
        s_full2_masked = self.s_full2[z2_mask]
        if len(s_full2_masked) == 0:
            print("Error. No solution. The whole function is equal to 0")
            return (None,None)
        lower_bound = -np.min(s_full2_masked)
        

        n = len(z2)

        x_max = np.max(np.sqrt(n) * z_mod - self.s_full2)

        interval_prec = mp.fsub(x_max, lower_bound)

    
        interval = float(interval_prec)

        if interval == 0.0:
            print("Error. Zero interval. Precision is too small")
            l = lower_bound
            return (None, None)
       

        # Scale the search interval to improve precision
        scaled_lower_bound = lower_bound / interval
        scaled_x_max = x_max / interval

        def f(l_t):
            return np.sum(z2 / ((self.s_full2 + (l_t + scaled_lower_bound)*interval) ** 2)) - 1

        # def df(l_t):
        #     return (-2) * interval * np.sum(z2 / ((self.s_full2 + (l_t + scaled_lower_bound)*interval) ** 3))

        # def dg(l_t):
        #     return 1/(2*np.sqrt(l_t + 1))

        new_max = scaled_x_max-scaled_lower_bound
        step = new_max
        while f(step) <= 0:
            step /= 2

        try:
            x, rep = brentq(f, step, new_max, full_output=True)
        except:
            print("Error. Brentq failed")
            return (None,None)

        l = (x + scaled_lower_bound) * interval

        if only_loss:

            loss = np.sum(b**2) - np.sum(z2 * ((self.s_full2 + 2*l) / ((self.s_full2 + l) ** 2 )))

            if take_mean:
                loss /= self.m

            return (loss, None)
        else:

            x = self.ridge(b,l)
            real_length = np.linalg.norm(x,2)

            x /= real_length

            loss = np.sum((np.dot(self.A, x) - b) ** 2)

            if take_mean:
                loss /= self.m

            return (loss, x) 
       
# TODO: Give credits to Yuanzhang
class UnitLstsqSDR:

    def __init__(self, A):
        self.A = np.asmatrix(A)
        self.m, self.n = A.shape

        self.cov_mat = np.transpose(A) @ A
    
    def solve(self,b,take_mean=True):

        if b is None:
            b = np.zeros((self.m,1))
        else:
            b = b.reshape(-1,1)

        b = np.asmatrix(b)

        # create a symmetric n-by-n matrix variable X = x * x^T
        X = cp.Variable((self.n, self.n), symmetric=True)
        x = cp.Variable((self.n, 1))

        # create the constraints
        constraints_SDR = [cp.vstack((cp.hstack((X, x)), cp.hstack((x.T, np.asmatrix(1))))) >> 0]
        constraints_SDR += [cp.trace(X) == 1]

        # solve the SDR
        prob_SDR = cp.Problem(cp.Minimize(cp.trace(self.cov_mat @ X) - 2 * (np.transpose(np.transpose(self.A) @ b)) @ x + np.transpose(b) @ b),
                            constraints_SDR)
        prob_SDR.solve(solver='CVXOPT')

        if prob_SDR.status != 'optimal':
            print("SDR failed")
            return (None,None)

        # use the randomization method to recover the solution to the original problem
        # number of random samples
        num_sample = 100
        # generate Gaussian random vectors with zero mean and covariance matrix X
        x_sample = np.random.multivariate_normal(x.value.flatten(), X.value, num_sample)

        x_sample_norm = np.linalg.norm(x_sample, axis=1)
        scale = 1./np.transpose(np.asmatrix(x_sample_norm))
        x_recovered = np.multiply(scale, x_sample)

        obj_recovered = np.linalg.norm(self.A@np.transpose(x_recovered)-b, axis=0)**2
        index_best = np.argmin(obj_recovered)
        x_best = x_recovered[index_best, :]

        # print result.
        # print("The best objective value recovered is", obj_recovered[index_best])
        # print("The best recovered solution is")
        # print(x_best)
        if take_mean:
            loss = obj_recovered[index_best] / self.m
        else:
            loss = obj_recovered[index_best]
        return (loss, x_best.flatten())

# TODO: GIVE CREDITS to Yuanzhang
class UnitLstsqKKT:

    def __init__(self, A):
        self.A = np.asmatrix(A)
        self.m, self.n = A.shape
        A_square = np.transpose(A) @ A
        lambdas, U = np.linalg.eigh(A_square)
        # sort the eigenvalues in the ascending order (numpy seems to have sorted them in this way; but just to make sure)
        index_eigenvalue_ascending = np.argsort(lambdas)
        self.lambdas = lambdas[index_eigenvalue_ascending]
        self.U = U[:, index_eigenvalue_ascending]

        self.pre_z = np.transpose(self.U) @ np.transpose(self.A)

        

    def solve(self, b, take_mean=True):

        b = b.reshape((-1,1))

        # the KKT conditions are:
        # 1. ( diag(lambdas) + mu * eye(n) ) * y = z, where z = U * A^T * b
        # 2. y^T * y = 1
        # where y is the primal solution and mu is the dual variable
        z = self.pre_z @ b

        """
        the solution can be written as y = inverse( diag(lambdas) + mu * eye(n) ) * z,
        where mu's are chosen such that y^T * y = 1
        there are at most (2n) mu's such that y^T * y = 1, and all mu's can be found using the bisection method
        """

        # define a function that calculate y^T * y for a given mu
        def norm_of_primal_variable(mu, lambdas, z):
            return sum(np.power(np.asarray(z).flatten() / (mu + lambdas), 2))


        # define a function that calculate the derivative of (y^T * y) with respect to mu
        def derivative_norm_of_primal_variable(mu, lambdas, z):
            return sum(-2 * np.power(np.asarray(z).flatten(), 2) / np.power((mu + lambdas), 3))


        # we first find the (n+1) intervals in which mu's can appear:
        lower_bound_mu = np.zeros((1, 1))
        upper_bound_mu = np.zeros((1, 1))
        ascending = np.zeros((1, 1))  # if the norm is ascending in mu
        # the first interval is (-lambda_1, infinity)
        lower_bound_mu[0, 0] = - self.lambdas[0] + 0.9 * np.absolute(z[0])
        upper_bound_mu[0, 0] = lower_bound_mu[0, 0]
        while norm_of_primal_variable(upper_bound_mu[0, 0], self.lambdas, z) >= 1:
            lower_bound_mu[0, 0] = upper_bound_mu[0, 0]
            upper_bound_mu[0, 0] = np.absolute(upper_bound_mu[0, 0]) * 2

        # the next 2*(n-1) possible intervals are (-lambda_{i+1}, mu_min_norm_i], [mu_min_norm_i, -lambda_i) for i=i,...,n-1, where mu_min_norm_i is the mu that results in minimum norm of the primal variable
        for index_interval in range(self.n-1):
            # this interval is between -lambda_{index+1} and -lambda_{index}
            # find mu that results in minimum (y^T y) when mu is in this interval
            lower_bound_mu_derivative = -self.lambdas[index_interval+1] + 1e-10
            upper_bound_mu_derivative = -self.lambdas[index_interval] - 1e-10

            mu_min_norm = (lower_bound_mu_derivative + upper_bound_mu_derivative) / 2
            while np.absolute(derivative_norm_of_primal_variable(mu_min_norm, self.lambdas, z)) > 1e-5:
                current_derivative = derivative_norm_of_primal_variable(mu_min_norm, self.lambdas, z)
                if current_derivative < 0:
                    lower_bound_mu_derivative = mu_min_norm
                else:
                    upper_bound_mu_derivative = mu_min_norm

                mu_min_norm = (lower_bound_mu_derivative + upper_bound_mu_derivative) / 2

            if norm_of_primal_variable(mu_min_norm, self.lambdas, z) < 1:
                lower_bound_mu = np.concatenate((lower_bound_mu, np.asmatrix(mu_min_norm)), 1)
                upper_bound_mu = np.concatenate((upper_bound_mu, - self.lambdas[index_interval] - 0.9 * np.absolute(z[index_interval])), 1)
                ascending = np.concatenate((ascending, np.asmatrix(1)), 1)

                lower_bound_mu = np.concatenate((lower_bound_mu, - self.lambdas[index_interval+1] + 0.9 * np.absolute(z[index_interval+1])), 1)
                upper_bound_mu = np.concatenate((upper_bound_mu, np.asmatrix(mu_min_norm)), 1)
                ascending = np.concatenate((ascending, np.asmatrix(0)), 1)

        # the last interval is (-infinity, -lambda_n)
        upper_bound_mu = np.concatenate((upper_bound_mu, - self.lambdas[self.n-1] - 0.9 * np.absolute(z[self.n-1])), 1)
        lower_bound_mu = np.concatenate((lower_bound_mu, np.asmatrix(upper_bound_mu[0, -1])), 1)
        while norm_of_primal_variable(lower_bound_mu[0, -1], self.lambdas, z) >= 1:
            upper_bound_mu[0, -1] = lower_bound_mu[0, -1]
            lower_bound_mu[0, -1] = - np.absolute(lower_bound_mu[0, -1]) * 2
        ascending = np.concatenate((ascending, np.asmatrix(1)), 1)

        # now we can find the mu in each interval
        mu = []
        for index_mu in range(np.shape(upper_bound_mu)[1]):
            current_lower_bound_mu = lower_bound_mu[0, index_mu]
            current_upper_bound_mu = upper_bound_mu[0, index_mu]

            current_mu = (current_lower_bound_mu + current_upper_bound_mu) / 2
            while np.absolute(norm_of_primal_variable(current_mu, self.lambdas, z) - 1.0) > 1e-5:
                current_norm = norm_of_primal_variable(current_mu, self.lambdas, z)
                if ascending[0, index_mu] == 0:
                    if current_norm < 1:
                        current_upper_bound_mu = current_mu
                    else:
                        current_lower_bound_mu = current_mu
                else:
                    if current_norm < 1:
                        current_lower_bound_mu = current_mu
                    else:
                        current_upper_bound_mu = current_mu

                current_mu = (current_lower_bound_mu + current_upper_bound_mu) / 2

            mu.append(current_mu)

        # finally, we can compute the primal solutions given each mu
        Y = z / (np.transpose(np.asmatrix(self.lambdas)) + np.asmatrix(mu))

        # we calculate the objective values under each solution and pick the solution with the minimum objective value
        obj = np.transpose(b) @ b - np.sum(np.multiply(np.power(Y, 2), np.transpose(np.asmatrix(self.lambdas))), 0) - 2 * np.asmatrix(mu)
        index_best = np.argmin(obj)
        y_best = Y[:, index_best]

        if take_mean:
            return (obj[0, index_best] / self.m, y_best)
        else:
            return (obj[0, index_best], y_best)

class UnitLstsqKKT_brent:

    def __init__(self, A):
        self.A = np.asmatrix(A)
        self.m, self.n = A.shape
        A_square = np.transpose(A) @ A
        lambdas, U = np.linalg.eigh(A_square)
        # sort the eigenvalues in the ascending order (numpy seems to have sorted them in this way; but just to make sure)
        index_eigenvalue_ascending = np.argsort(lambdas)
        self.lambdas = lambdas[index_eigenvalue_ascending]
        self.U = U[:, index_eigenvalue_ascending]

        self.pre_z = np.transpose(self.U) @ np.transpose(self.A)

    def solve(self, b, take_mean=True):

        if b is None:
            b = np.zeros((self.m,1))
        else:
            b = b.reshape(-1,1)

        # the KKT conditions are:
        # 1. ( diag(lambdas) + mu * eye(n) ) * y = z, where z = U * A^T * b
        # 2. y^T * y = 1
        # where y is the primal solution and mu is the dual variable
        z = self.pre_z @ b

        """
        the solution can be written as y = inverse( diag(lambdas) + mu * eye(n) ) * z,
        where mu's are chosen such that y^T * y = 1
        there are at most (2n) mu's such that y^T * y = 1, and all mu's can be found using the bisection method
        """

        # define a function that calculate y^T * y for a given mu
        def f(mu):
            return sum(np.power(np.asarray(z).flatten() / (mu + self.lambdas), 2)) - 1

        # define a function that calculate the derivative of (y^T * y) with respect to mu
        def derivative_f(mu):
            return sum(-2 * np.power(np.asarray(z).flatten(), 2) / np.power((mu + self.lambdas), 3))

        # we first find the (n+1) intervals in which mu's can appear:
        lower_bound_mu = np.zeros((1, 1))
        upper_bound_mu = np.zeros((1, 1))
        # the first interval is (-lambda_1, infinity)
        lower_bound_mu[0, 0] = - self.lambdas[0] + 0.9 * np.absolute(z[0])
        upper_bound_mu[0, 0] = lower_bound_mu[0, 0]
        while f(upper_bound_mu[0, 0]) >= 0:
            lower_bound_mu[0, 0] = upper_bound_mu[0, 0]
            upper_bound_mu[0, 0] = np.absolute(upper_bound_mu[0, 0]) * 2

        # the next 2*(n-1) possible intervals are (-lambda_{i+1}, mu_min_norm_i], [mu_min_norm_i, -lambda_i) for i=i,...,n-1, where mu_min_norm_i is the mu that results in minimum norm of the primal variable
        for index_interval in range(self.n - 1):
            # this interval is between -lambda_{index+1} and -lambda_{index}
            # find mu that results in minimum (y^T y) when mu is in this interval
            lower_bound_mu_derivative = -self.lambdas[index_interval + 1] + 1e-4
            upper_bound_mu_derivative = -self.lambdas[index_interval] - 1e-4

            try:
                mu_min_norm, _ = brentq(derivative_f, lower_bound_mu_derivative, upper_bound_mu_derivative, full_output=True)
            except:
                print("Error. Brentq failed")
                return (None, None)

            if f(mu_min_norm) < 0:
                lower_bound_mu = np.concatenate((lower_bound_mu, np.asmatrix(mu_min_norm)), 1)
                upper_bound_mu = np.concatenate(
                    (upper_bound_mu, - self.lambdas[index_interval] - 0.9 * np.absolute(z[index_interval])), 1)

                lower_bound_mu = np.concatenate(
                    (lower_bound_mu, - self.lambdas[index_interval + 1] + 0.9 * np.absolute(z[index_interval + 1])), 1)
                upper_bound_mu = np.concatenate((upper_bound_mu, np.asmatrix(mu_min_norm)), 1)

        # the last interval is (-infinity, -lambda_n)
        upper_bound_mu = np.concatenate((upper_bound_mu, - self.lambdas[self.n - 1] - 0.9 * np.absolute(z[self.n - 1])),
                                        1)
        lower_bound_mu = np.concatenate((lower_bound_mu, np.asmatrix(upper_bound_mu[0, -1])), 1)
        while f(lower_bound_mu[0, -1]) >= 0:
            upper_bound_mu[0, -1] = lower_bound_mu[0, -1]
            lower_bound_mu[0, -1] = - np.absolute(lower_bound_mu[0, -1]) * 2

        # now we can find the mu in each interval
        mu = []
        for index_mu in range(np.shape(upper_bound_mu)[1]):
            current_lower_bound_mu = lower_bound_mu[0, index_mu]
            current_upper_bound_mu = upper_bound_mu[0, index_mu]

            try:
                current_mu, _ = brentq(f, current_lower_bound_mu, current_upper_bound_mu, full_output=True)
            except:
                print("Error. Brentq failed")
                return (None, None)

            mu.append(current_mu)

        # finally, we can compute the primal solutions given each mu
        Y = z / (np.transpose(np.asmatrix(self.lambdas)) + np.asmatrix(mu))

        # we calculate the objective values under each solution and pick the solution with the minimum objective value
        obj = np.transpose(b) @ b - np.sum(np.multiply(np.power(Y, 2), np.transpose(np.asmatrix(self.lambdas))),
                                           0) - 2 * np.asmatrix(mu)

        # filter out zero vectors
        norms = np.linalg.norm(Y,axis=0)
        obj_filtered = obj[:, (norms != 0)]
        if obj_filtered[0,:].size == 0:
            print("KKT-brent failed")
            return (None, None)
        index_best = np.argmin(obj_filtered)
        y_best = (Y[:, (norms != 0)])[:, index_best]
      
        if index_best != 0:
            print(f"index: {index_best}")
  
        if take_mean:
            return (obj[0, index_best] / self.m, np.ravel(y_best))
        else:
            return (obj[0, index_best], np.ravel(y_best))


class UnitLstsqLARS:

    def __init__(self,A):
        
        self.A_T = A.T
        self.A = A
        self.gram = A.T @ A
        self.m, self.n = A.shape
        
    
    def solve(self,b,take_mean=True):

        if b is None:
            Xy = np.zeros(self.n,)
        else:
            Xy = np.dot(self.A_T,b)

        alphas, _, coefs = lars_path_gram(Xy=Xy, Gram=self.gram, n_samples=self.m, method='lasso')

        norms = np.sum(np.abs(coefs),axis=0)

        index = bisect_left(norms,1.0)

        if index == len(norms):
            weights = coefs[:,-1] / norms[-1]
        elif norms[index] == 1.0:
            weights = coefs[:,index]
        else:
            coefs_start = coefs[:,index]
            coefs_end = coefs[:,index-1]
            t_start = alphas[index]
            t_end = alphas[index-1]
            delta_t = t_end - t_start
    
            def f(t):
                return np.sum(np.abs((t-t_start)*(coefs_end - coefs_start)/delta_t + coefs_start)) - 1
        
            result, rep = brentq(f, t_start, t_end, full_output=True)
            weights = (result-t_start)*(coefs_end - coefs_start)/delta_t + coefs_start

        if take_mean:
            loss = np.sum(np.power(np.dot(self.A,weights) - b,2)) / self.m
        else:
            loss = np.sum(np.power(np.dot(self.A,weights) - b,2)) 

        return (loss,weights)

class UnitLstsqLARSImproved:

    def __init__(self,A):
        
        self.A_T = A.T
        self.A = A
        self.gram = A.T @ A
        self.m, self.n = A.shape

        self.cvx = UnitL1NormLeastSquare_CVX(A)
        self.homogeneous_loss, self.homogeneous_solution = self.cvx.solve(np.zeros((self.m,1)),take_mean=False)
        
    
    def solve(self,b,take_mean=True):

        if b is None:
            if take_mean:
                loss = self.homogeneous_loss / self.m
            else:
                loss = self.homogeneous_loss
            return (loss, self.homogeneous_solution)
        else:
            Xy = np.dot(self.A_T,b)

        alphas, _, coefs = lars_path_gram(Xy=Xy, Gram=self.gram, n_samples=self.m, method='lasso')

        norms = np.sum(np.abs(coefs),axis=0)

        index = bisect_left(norms,1.0)

        if index == len(norms):
            if norms[-1] == 0.0:
                # We have a homogeneous case
                if take_mean:
                    loss = self.homogeneous_loss / self.m
                else:
                    loss = self.homogeneous_loss
                return (loss, self.homogeneous_solution)

            coefs_start = coefs[:,-1]
            coefs_end = coefs[:,-2]
            t_start = alphas[-1]
            t_end = alphas[-2]
            delta_t = t_end - t_start

            def f(t):
                return np.sum(np.abs((t-t_start)*(coefs_end - coefs_start)/delta_t + coefs_start)) - 1
            t_optim_end = t_start # it should be 0 but sometimes it's just very close to 0

            differences = coefs_end - coefs_start
            values_at_0 = coefs_start

            mask = (differences * values_at_0) > 0.0
            if np.sum(mask) == 0:
                ground_zero = 0.0
            else:
                ground_zero = - np.max((values_at_0[mask] * delta_t) / differences[mask])
            # from ground_zero the norm can only increase (when alpha goes more negative)
            if f(ground_zero) == 0.0:
                result = ground_zero
            elif f(ground_zero) > 0:
                t_optim_start = ground_zero
                result, rep = brentq(f, t_optim_start, t_optim_end, full_output=True)
            else:
                abs_slopes_sum = np.sum(np.abs((coefs_end - coefs_start)/delta_t))
                result = ground_zero + f(ground_zero) / abs_slopes_sum
           
            weights = (result-t_start)*(coefs_end - coefs_start)/delta_t + coefs_start

        elif norms[index] == 1.0:
            weights = coefs[:,index]
        else:
            coefs_start = coefs[:,index]
            coefs_end = coefs[:,index-1]
            t_start = alphas[index]
            t_end = alphas[index-1]
            delta_t = t_end - t_start
    
            def f(t):
                return np.sum(np.abs((t-t_start)*(coefs_end - coefs_start)/delta_t + coefs_start)) - 1
        
            result, rep = brentq(f, t_start, t_end, full_output=True)
            weights = (result-t_start)*(coefs_end - coefs_start)/delta_t + coefs_start

        if take_mean:
            loss = np.sum(np.power(np.dot(self.A,weights) - b,2)) / self.m
        else:
            loss = np.sum(np.power(np.dot(self.A,weights) - b,2)) 

        return (loss,weights)


# class UnitLstsqPGD:
#     def __init__(self,A):
#         self.A = torch.from_numpy(A).float()
#         self.m, self.n = A.shape
#         self.model = UnitLstsqPGD.LinearRegressionModel(self.n)
#         self.criterion = torch.nn.MSELoss()
#         self.clipper = UnitLstsqPGD.UnitNormClipper(1)
#         self.num_epochs = 100

#     def weights_init(m):
#         if type(m) == torch.nn.Linear:
#             torch.nn.init.xavier_normal_(m.weight.data)

#     def solve(self,b,take_mean=True):
#         self.model.apply(UnitLstsqPGD.weights_init)
#         optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
#         b = torch.from_numpy(b).float()
        
#         for epoch in range(self.num_epochs):
#             pred_y = self.model(self.A).view(-1)
#             loss = self.criterion(pred_y,b)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if epoch % self.clipper.frequency == 0:
#                 self.model.apply(self.clipper)

#         weights = self.model.linear.state_dict()['weight'].view(-1)
#         loss = torch.sum(torch.pow(torch.matmul(self.A, weights) - b,2))

#         if take_mean:
#             loss /= self.m

#         return (loss,weights.numpy())

#     class LinearRegressionModel(torch.nn.Module):

#         def __init__(self,n):
#             super(UnitLstsqPGD.LinearRegressionModel, self).__init__()
#             self.linear = torch.nn.Linear(n,1,bias=False)
        
#         def forward(self, x):
#             y_pred = self.linear(x)
#             return y_pred

#     class UnitNormClipper(object):

#         def __init__(self, frequency=5):
#             self.frequency = frequency

#         def __call__(self, module):
#             # filter the variables to get the ones you want
#             if hasattr(module, 'weight'):
#                 w = module.weight.data
#                 w.div_(torch.linalg.vector_norm(w, 1).expand_as(w))
        
class UnitLstsqMD:
    def __init__(self,A):
        self.A = A
        self.A_T = A.T
        self.gram = self.A_T @ self.A
        self.m, self.n = A.shape
        self.orthants = list(product([-1,1],repeat=self.n))
        self.gram_norm = np.max(np.linalg.norm(self.gram,np.inf,axis=0))
        self.A_norm = np.max(np.linalg.norm(self.A,2,axis=0))

       

    def solve(self,b,take_mean=True):
        if b is None:
            b = np.zeros(self.m)

        c = np.dot(self.A_T,b)
        c_inf = np.linalg.norm(c,np.inf)
        l_constant = 2*(c_inf + self.gram_norm)

        def glob_f(x):
            return np.sum(np.power(np.dot(self.A,x)-b,2))
        def glob_df(x):
            return 2*(np.dot(self.gram,x)-c)

        f_values = []
        df_values = []
        for orth in self.orthants:
            f_values.append(glob_f(np.array(orth)))
            df_values.append(glob_df(np.array(orth)))
        
        max_f_value = np.max(f_values)
        df_norms = [np.linalg.norm(d,ord=np.inf) for d in df_values]
        max_df_norm = np.max(df_norms)

        sols = []
        for orth in self.orthants:
            
            orth = np.array(orth)
            def f(x):
                return np.sum(np.power(np.dot(self.A,x*orth)-b,2))
            def df(x):
                return 2*(np.dot(self.gram,x*orth)-c)*orth
            md = MirrorDescentSimplex(self.n,df)
            
            # diameter = self.A_norm ** 2 + 2*c_inf + np.sum(b ** 2) - f(md.x)
            diameter = max_f_value - f(md.x)
            sol = md.optimize(20,20/max_df_norm)
            sols.append(sol*orth)
        
       
        losses = [glob_f(sol) for sol in sols]
        try:
            index = np.argmin(losses)
        except:
            print("Empty list")
            return (None,None)
        weights = sols[index]
        loss = losses[index]

        if take_mean:
            loss /= self.m
        
        return (loss,weights)


class UnitL1NormLeastSquare_CVX:
    # this function solves 2^n optimization problems using CVX and picks the minimal loss value
    # this approach will give us the optimal solution

    def __init__(self, A):
        self.A = np.asmatrix(A)
        self.m, self.n = A.shape

        self.cov_mat = np.transpose(A) @ A

    def solve(self, b, take_mean=True):

        if b is None:
            b = np.zeros((self.m, 1))
        else:
            b = b.reshape(-1, 1)

        b = np.asmatrix(b)

        num_simplex = 2 ** self.n
        obj_candidate = np.zeros((1, num_simplex))
        x_candidate = np.zeros((self.n, num_simplex))
        for index_simplex in range(num_simplex):
            simplex_representation_string = f'{index_simplex:0{self.n}b}'  # a string of the binary representation
            simplex_representation_array = np.array(
                [int(x) for x in simplex_representation_string])  # convert the string to a numeric array
            simplex_representation_array = 2 * simplex_representation_array - 1  # map 0's to -1's
            D = np.asmatrix(np.diag(simplex_representation_array))  # define the diagonal matrix D

            # define the variable
            y = cp.Variable((self.n, 1))

            # create the constraints
            constraints = [y >= 0, cp.sum(y) == 1]

            # solve the optimization problem
            prob = cp.Problem(cp.Minimize(cp.norm(self.A @ D @ y - b)),
                              constraints)
            prob.solve()

            if prob.status != 'optimal':
                print("CVX failed")
                return (None, None)

            obj_candidate[0, index_simplex] = prob.value
            x_candidate[:, index_simplex] = simplex_representation_array * np.transpose(y.value)

        index_best = np.argmin(obj_candidate)
        x_best = x_candidate[:, index_best]

        #print(f"signs of the optimal solution: {np.transpose(np.sign(x_best))}")

        # print result.
        # print("The best objective value recovered is", obj_recovered[index_best])
        # print("The best recovered solution is")
        # print(x_best)
        if take_mean:
            obj_best = obj_candidate[0, index_best] ** 2 / self.m
        else:
            obj_best = obj_candidate[0, index_best] ** 2
        return obj_best, x_best.flatten()


class UnitL1NormLeastSquare_CVX_heuristic:
    # solves the standard least square problem (i.e., no constraint)
    # solves one optimization problem, which solution has the same signs as the solution to the standard least square problem

    def __init__(self, A):
        self.A = np.asmatrix(A)
        self.m, self.n = A.shape

        self.cov_mat = np.transpose(A) @ A
        self.inverse_cov_mat = np.linalg.inv(self.cov_mat)

    def solve(self, b, take_mean=True):

        if b is None:
            b = np.zeros((self.m, 1))
        else:
            b = b.reshape(-1, 1)

        b = np.asmatrix(b)

        # define the variable
        x = self.inverse_cov_mat @ np.transpose(self.A) @ b

        '''
        # solve the optimization problem
        prob = cp.Problem(cp.Minimize(cp.norm(self.A @ x - b)))
        prob.solve()
        '''

        print(f"L-1 norm of least square: {np.sum(np.absolute(x))}")

        sign_x = np.sign(np.array(x))
        sign_x[sign_x == 0] = 1
        D = np.asmatrix(np.diag(sign_x.reshape(self.n, )))

        print(f"signs of the least square solution: {np.transpose(sign_x)}")
        print(f"signs of inv(A^T*A)*sign(x): {np.transpose(np.sign(self.inverse_cov_mat @ sign_x))}")

        # define the variable
        y = cp.Variable((self.n, 1))

        # create the constraints
        constraints = [y >= 0, cp.sum(y) == 1]

        # solve the optimization problem
        prob = cp.Problem(cp.Minimize(cp.norm(self.A @ D @ y - b)),
                          constraints)
        prob.solve()

        if prob.status != 'optimal':
            print("CVX failed")
            return (None, None)

        if take_mean:
            obj = prob.value ** 2 / self.m
        else:
            obj = prob.value ** 2
        return obj, y.value.flatten()


class UnitL1NormLeastSquare_heuristic:
    # finds a solution when the L-1 norm of the standard least square solution is smaller than 1
    # in this case, we need \lambda < 0
    # keep the same signs as the solution to the standard least square problem
    # since the L-1 norm is strictly decreasing in \lambda, the solution can be found efficiently by Brent's method

    def __init__(self, A):
        self.A = np.asmatrix(A)
        self.m, self.n = A.shape

        self.cov_mat = np.transpose(A) @ A
        self.inverse_cov_mat = np.linalg.inv(self.cov_mat)

    def solve(self, b, standard_least_square, take_mean=True):

        sign_x = np.sign(np.array(standard_least_square))
        sign_x[sign_x == 0] = 1
        y = self.inverse_cov_mat @ sign_x

        def f(t):
            return np.sum(np.abs(standard_least_square - t * y)) - 1

        lower_bound = np.max((standard_least_square - sign_x) / y)
        dual_variable, _ = brentq(f, lower_bound, 0.0, full_output=True)

        x = standard_least_square - dual_variable * y

        if take_mean:
            obj = np.linalg.norm(self.A @ x - b) ** 2 / self.m
        else:
            obj = np.linalg.norm(self.A @ x - b) ** 2
        return obj, x.flatten()


class UnitLstsqHeuristicFull:

    def __init__(self,A):
        self.A = np.asmatrix(A)
        self.m, self.n = A.shape

        self.pre_standard_least_squares = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A)
        self.heuristic_solver =  UnitL1NormLeastSquare_heuristic(A)
        self.lars_solver = UnitLstsqLARS(A)


    def solve(self, b, take_mean=True):
        
        if b is None:
            b = np.zeros((self.m, 1))
        else:
            b = b.reshape(-1, 1)

        standard_least_square = self.pre_standard_least_squares @ b
        L1_norm = np.sum(np.absolute(standard_least_square))
        print(L1_norm)
        if L1_norm == 1.0:
            print("Standard LSTSQ")
            loss = np.linalg.norm(self.A @ standard_least_square - b)**2
            if take_mean:
                loss /= self.m
            x = standard_least_square
        elif L1_norm < 1.0:
            print("Heuristic")
            loss, x = self.heuristic_solver.solve(b, standard_least_square, take_mean=take_mean)
        else:
            print("LARS")
            b = b.reshape(-1, )
            loss, x = self.lars_solver.solve(b, take_mean=take_mean)

        return loss, x





if __name__ == "__main__":


    np.random.seed(0)

    # num_tests = 1
    # scale_factor  = 100 # The bigger the scale_factor the less uniform entries are
    # m = 10000
    # n = 5

    file_names = glob.glob("results/matrices/*.p")

    df = pd.DataFrame()

    for i, file_name in enumerate(file_names):

        if i == 2000:
            break
        print("-"*10)
        print(f"Test {i+1}/{len(file_names)}")

        # b = b.reshape(-1,1)

        # solver1 = UnitLstsqSDR(A)
        # start = time.time()
        # loss1,x1 = solver1.solve(b, take_mean=False)
        # end = time.time()  
        # if loss1 is not None:
        #     print(f"SDR | Loss: {loss1} | Time: {end-start} seconds")

        # solver2 = UnitLstsqSVD(A)
        # b=b.reshape(-1,)
        # start = time.time()
        # loss2,x2 = solver2.solve(b, take_mean=False)
        # end = time.time()
        # if loss2 is not None:
        #     print(f"SVD | Loss: {loss2} | Time: {end-start} seconds")
        #     print(x2)

        # solver3 = UnitLstsqKKT(A)
        # b = b.reshape(-1, 1)
        # start = time.time()
        # try:
        #     loss3, x3 = solver3.solve(b, take_mean=False)
        #     end = time.time()
        #     print(f"KKT | Loss {loss3} | Time: {end-start} seconds")
        # except:
        #     print("KKT failed")
       

        # solver4 = UnitLstsqKKT_brent(A)
        # b = b.reshape(-1, 1)
        # start = time.time()
        # try:
        #     loss4, x4 = solver4.solve(b, take_mean=False)
        #     end = time.time()
        #     print(f"KKT-Brent | Loss {loss4} | Time: {end - start} seconds")
        # except:
        #     print("KKT-Brent failed")

        with open(file_name,'rb') as file:
            problem = pickle.load(file)
        A = problem['X']
        b = problem['b']

        record = {}

    
        solver5 = UnitLstsqLARS(A)
        b = b.reshape(-1,)
        start = time.time()
        try: 
            loss5, x5 = solver5.solve(b, take_mean=False)
            end = time.time()
            print(f"LARS | Loss {loss5} | Time: {end - start} seconds")
            print(x5)
        except:
            loss5 = np.nan
            print("LARS failed")
        record['lars_loss'] = loss5
        record['lars_time'] = end - start


        solver6 = UnitLstsqMD(A)
        b = b.reshape(-1,)
        start = time.time()
        try: 
            loss6, x6 = solver6.solve(b, take_mean=False)
            end = time.time()
            print(f"MD | Loss {loss6} | Time: {end - start} seconds")
            print(x6)
        except:
            loss6 = np.nan
            print("MD failed")
        record['md_loss'] = loss6
        record['md_time'] = end - start

        solver7 = UnitL1NormLeastSquare_CVX(A)
        start = time.time()
        loss7, x7 = solver7.solve(b, take_mean=False)
        end = time.time()
        if loss7 is not None:
            print(f"CVX | Loss: {loss7} | Time: {end - start} seconds")
            print(x7)
        else:
            loss7 = np.nan
        record['cvx_loss'] = loss7
        record['cvx_time'] = end - start

        solver8 = UnitL1NormLeastSquare_CVX_heuristic(A)
        start = time.time()
        loss8, x8 = solver8.solve(b, take_mean=False)
        end = time.time()
        if loss8 is not None:
            print(f"CVX-heuristic | Loss: {loss8} | Time: {end - start} seconds")
            print(x8)
        else:
            loss8 = np.nan
        record['cvx_heur_loss'] = loss8
        record['cvx_heur_time'] = end - start

        solver9 = UnitLstsqHeuristicFull(A)
        b = b.reshape(-1,1)
        start = time.time()
        loss9, x9 = solver9.solve(b, take_mean=False)
        end = time.time()
        if loss9 is not None:
            print(f"heuristic | Loss: {loss9} | Time: {end - start} seconds")
            print(x9)
        else:
            loss9 = np.nan
        record['heur_loss'] = loss9
        record['heur_time'] = end - start

        solver10 = UnitLstsqLARSImproved(A)
        b = b.reshape(-1,)
        start = time.time()
        try: 
            loss10, x10 = solver10.solve(b, take_mean=False)
            end = time.time()
            print(f"LARS_Improved | Loss {loss10} | Time: {end - start} seconds")
            print(x10)
        except:
            loss10 = np.nan
            print("LARS improved failed")
        record['lars_imp_loss'] = loss10
        record['lars_imp_time'] = end - start


        b = b.reshape(-1,1)
        standard_least_square = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A) @ b
        L1_norm = np.sum(np.absolute(standard_least_square))
        record['L1_norm_lstsq'] = L1_norm

        df = df.append(record,ignore_index=True)

    df.to_csv('results/comparison.csv')
    

