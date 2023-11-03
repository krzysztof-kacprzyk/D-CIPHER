import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, brentq
import cvxpy as cp
import time

from sklearn.linear_model import lars_path_gram
from bisect import bisect_left
import mpmath as mp
from itertools import product
import glob
import pandas as pd

def ridge(A,b,l):
    n = A.shape[1]
    x = np.dot(np.linalg.inv((np.transpose(A) @ A) + l*np.eye(n)) @ np.transpose(A),b)
    return x


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

            if 'optimal' not in prob.status:
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
    

class UnitLstsqLARSImproved:

    def __init__(self,A):
        
        self.A_T = A.T
        self.A = A
        self.gram = A.T @ A
        self.m, self.n = A.shape

        self.homogeneous_calculated = False
    
    def calculate_homogeneous(self):
        self.cvx = UnitL1NormLeastSquare_CVX(self.A)
        self.homogeneous_loss, self.homogeneous_solution = self.cvx.solve(np.zeros((self.m,1)),take_mean=False)
        self.homogeneous_calculated = True

    
    def solve(self,b,take_mean=True):

        if b is None:
            if not self.homogeneous_calculated:
                self.calculate_homogeneous()
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
                if not self.homogeneous_calculated:
                    self.calculate_homogeneous()
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



if __name__ == "__main__":

    np.random.seed(0)

    num_tests = 1000
    scale_factor  = 100 # The bigger the scale_factor the less uniform entries are
    m = 1000

    for n in range(1,8):

        df = pd.DataFrame()

        for i in range(num_tests):

            print("-"*10)
            print(f"Test {i+1}/{num_tests}")

            A = np.random.normal(0.0,1.0,(m,n))
            b = np.random.normal(0.0,1.0,(m,1))
        

            # We want the matrix A and vector b to have entries from widely different scales
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if np.random.rand() > 0.5:
                        A[i,j] /= scale_factor
                    else:
                        A[i,j] *= scale_factor
            
            for i in range(b.shape[0]):
                if np.random.rand() > 0.5:
                    b[i,0] /= scale_factor
                else:
                    b[i,0] *= scale_factor


            

            record = {}


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

            df = pd.concat([df,pd.DataFrame([record])],ignore_index=True)

        df.to_csv(f'experiments/results/CoLLie/comparison_n{n}.csv')
    

