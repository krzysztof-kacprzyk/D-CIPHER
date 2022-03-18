import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, brentq
from sympy import re
import cvxpy as cp
import time

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
            return None
        lower_bound = -np.min(s_full2_masked)
        

        n = len(z2)

        x_max = np.max(np.sqrt(n) * z_mod - self.s_full2)
    
        interval = (x_max - lower_bound)

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
       

class UnitLstsqSDR:

    def __init__(self, A):
        self.A = np.asmatrix(A)
        self.m, self.n = A.shape

        self.cov_mat = np.transpose(A) @ A
    
    def solve(self,b,take_mean=True):

        if b is None:
            b = np.zeros(self.m,1)
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



if __name__ == "__main__":


    np.random.seed(0)

    A = np.random.uniform(0.0,1.0,(1000,10))
    b = np.random.uniform(0.0,1.0,(1000,1))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.random.rand() > 0.5:
                A[i,j] /= 1000000000
            else:
                A[i,j] *= 1000000000
    
    for i in range(b.shape[0]):
        if np.random.rand() > 0.5:
            b[i,0] /= 1000000000
        else:
            b[i,0] *= 1000000000
    print(b)
    b = b.reshape(-1,1)

    solver1 = UnitLstsqSDR(A)
    start = time.time()
    loss1,x1 = solver1.solve(b)
    end = time.time()
    print(f"SDR took {end-start} seconds")
    print(loss1)
    print(np.linalg.norm(x1,2))

    solver2 = UnitLstsqSVD(A)
    b=b.reshape(-1,)
    start = time.time()
    loss2,x2 = solver2.solve(b,only_loss=False,take_mean=False)
    end = time.time()
    print(f"SVD took {end-start} seconds")
    print(loss2)
    print(np.linalg.norm(x2,2))

