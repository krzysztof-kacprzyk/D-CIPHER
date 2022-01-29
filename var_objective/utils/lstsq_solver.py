import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, brentq

def ridge(A,b,l):
    n = A.shape[1]
    x = np.dot(np.linalg.inv((np.transpose(A) @ A) + l*np.eye(n)) @ np.transpose(A),b)
    return x


class UnitLstsqSVD:

    def __init__(self, A):
        self.A = A

        m, n = self.A.shape

        u,s,vh = np.linalg.svd(A, full_matrices=True)

        s_full = np.zeros(n)
        for i,si in enumerate(s):
            s_full[i]=si

        p = s.shape[0]

        self.main_part = np.transpose(u[:,:p] * s) 

        self.s_full2 = s_full ** 2


    def solve(self,b,tol=0.01,only_loss=False,take_mean=True):

        np.set_printoptions(precision=20)

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
        else:

            # Scale the search interval to improve precision
            scaled_lower_bound = lower_bound / interval
            scaled_x_max = x_max / interval

            def f(l_t):
                return np.sum(z2 * (1 / (self.s_full2 + l_t*interval) ** 2)) - 1

            step = 1.0
            while f(scaled_lower_bound + step) < 0:
                step /= 2

            x, rep = brentq(f, scaled_lower_bound+step, scaled_x_max, full_output=True)
            l = x * interval

        if only_loss:

            loss = np.sum(b**2) - np.sum(z2 * ((self.s_full2 + 2*l) / ((self.s_full2 + l) ** 2 )))

            if take_mean:
                loss /= len(b)

            return (loss, None)
        else:

            x = ridge(self.A,b,l)
            length = np.linalg.norm(x,2)
            
            if np.abs(length - 1.0) > tol:
                print(f"Error. Something wrong with optimizer. Length found is {length}")
                return (None, None)

            if length == 0:
                print("Error. The solution has length 0")
                return (None, None)
            
            loss = np.sum((np.dot(self.A, x) - b) ** 2)

            if take_mean:
                loss /= len(b)

            return (loss, x) 
       


if __name__ == "__main__":

    np.random.seed(1)

    A = np.random.normal(0.0,1.0,(1000,10))
    b = np.random.normal(0.0,1.0,1000)


