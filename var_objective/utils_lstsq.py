import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, brentq

def plot_length(A,b,t):
    m = A.shape[0]
    n = A.shape[1]
    x = np.dot(np.linalg.inv((np.transpose(A) @ A) + t*np.eye(n)) @ np.transpose(A),b)
    length = np.sum(x ** 2)
    return length

def ridge(A,b,l):
    n = A.shape[1]
    x = np.dot(np.linalg.inv((np.transpose(A) @ A) + l*np.eye(n)) @ np.transpose(A),b)
    return x

def unit_lstsq_naive(A,b,iters=100,l=0.0, tol=0.001, min_l=-1000.0,max_l=1000.0):

    m = A.shape[0]
    n = A.shape[1]

    x = np.dot(np.linalg.inv((np.transpose(A) @ A)) @ np.transpose(A),b)
    length = np.sum(x ** 2)
    print(length)
    if np.abs(length - 1) < tol:
        print(f"Stopped at the beginning")
        return x


    left_l = min_l
    right_l = max_l

    for i in range(iters):
        x = np.dot(np.linalg.inv((np.transpose(A) @ A) + l*np.eye(n)) @ np.transpose(A),b)
        length = np.sum(x ** 2)
        print(l,length)
        if np.abs(length - 1) < tol:
            print(f"Stopped after {i+1} iterations")
            break
        else:
            if length < 1:
                right_l = l
                l = (l + left_l) / 2
            else:
                left_l = l
                l = (l + right_l) / 2
                
    return x

def unit_lstsq_svd(A,b,offset=1.0):

    f, jac, lb = create_f_and_jac(A,b)
    l = minimize(f, lb+offset, method='BFGS', jac=jac).x
    x = ridge(A,b,l)

    return x

def create_f_and_jac(A,b):
    m, n = A.shape

    u,s,vh = np.linalg.svd(A, full_matrices=True)

    s_full = np.zeros(n)
    for i,si in enumerate(s):
        s_full[i]=si

    p = s.shape[0]
   
    z = np.dot(np.transpose(u[:,:p] * s),b)

    z2 = z ** 2
    s_full2 = s_full ** 2
    
    def f(l):
        return (np.sum(z2 * (1 / (s_full2 + l) ** 2)) - 1) ** 2

    def jac(l):
        return 2 * (np.sum(z2 * (1 / (s_full2 + l) ** 2)) - 1) * np.sum(z2 * (-2) * ((s_full2 + l) ** (-3)))

    s2_min = np.min(s ** 2)

    lower_bound = -s2_min
    
    return f, jac, lower_bound


def projective_lstsq_affine_part(A,b):
    m, n = A.shape

    if n == 1:
        return np.array([1.0])
    
    A_last_column = A[:,-1]
    A_ = A[:,:-1]
    A_T = np.transpose(A)
    x_ = np.dot(np.linalg.pinv(A_T @ A_) @ A_T, b-A_last_column)
    x = np.concatenate([x_, np.ones(1)])
    return x
    
def projective_lstsq(A,b):
    m, n = A.shape

    sols = []

    for i in range(n):

        A_part = A[:,:(n-i)]
        x_part = projective_lstsq_affine_part(A_part,b)
        x = np.concatenate([x_part, np.zeros(i)])
        sols.append(x)

    losses = [np.sum((np.dot(A,x)-b) ** 2) for x in sols]
    print(losses)
    print(sols)
    min_i = np.argmin(losses)
    return (sols[min_i], losses[min_i])


class UnitLstsqSVD:

    def __init__(self, A, offset=0.1):
        self.A = A
        self.offset = offset

        m, n = self.A.shape

        u,s,vh = np.linalg.svd(A, full_matrices=True)

        s_full = np.zeros(n)
        for i,si in enumerate(s):
            s_full[i]=si

        p = s.shape[0]

        self.main_part = np.transpose(u[:,:p] * s) 

        self.s_full2 = s_full ** 2

        # s2_min = np.min(s ** 2)

        # self.lower_bound = -s2_min

        
        

    def solve(self,b,verbose=False,tol=0.01):

        np.set_printoptions(precision=20)

        z = np.dot(self.main_part,b)
        z2 = z ** 2

        z2_max = np.max(z2)

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
            print("Zero interval. Probably the precision is too small")
            # print(z_mod)
            # print(self.s_full2)
            # print(x_max)
            # print(lower_bound)
            # print(lower_bound * 1000000000000)
            l = lower_bound

            # x_max = np.max(1000000000000*np.sqrt(n) * z_mod - 1000000000000*self.s_full2)
            # print(x_max)
            # print(x_max - lower_bound * 1000000000000)
            # print(lower_bound * 1000000000000)
        else:

            scaled_lower_bound = lower_bound / interval
            scaled_x_max = x_max / interval

            def f(l_t):
                return np.sum(z2 * (1 / (self.s_full2 + l_t*interval) ** 2)) - 1

            def jac(l):
                return 2 * (np.sum(z2 * (1 / (self.s_full2 + l) ** 2)) - 1) * np.sum(z2 * (-2) * ((self.s_full2 + l) ** (-3)))



        
        
            step = 1.0
            while f(scaled_lower_bound + step) < 0:
                step /= 2

            x, rep = brentq(f, scaled_lower_bound+step, scaled_x_max, full_output=True)
            l = x * interval

        # rep = minimize_scalar(f, bounds=(scaled_lower_bound, scaled_x_max),method='bounded',options={'xatol':tol**3})
        # l = rep.x * interval


        x = ridge(self.A,b,l)
        length = np.linalg.norm(x,2)
        
        if np.abs(length - 1.0) > tol:
            print(f"Something wrong with optimizer. Length found is {length}")
            # if interval != 0:
                # print(f"Parameter is {l / interval}")
                #print(f"Delta is {f(l/interval)}")
                # time = np.linspace(scaled_lower_bound + step, scaled_x_max,1000)
                # plt.plot(time, [f(t) for t in time])
                # plt.show()
        if length == 0:
            return None
        return x / length
       


    


if __name__ == "__main__":

    np.random.seed(1)

    A = np.random.normal(0.0,1.0,(1000,10))
    b = np.random.normal(0.0,1.0,1000)

    
    # f, jac, lb = create_f_and_jac(A,b)

    # def f2(t):
    #     return (np.sum(np.dot(np.linalg.inv(np.transpose(A)@A + t*np.eye(A.shape[1])) @ np.transpose(A),b) ** 2) - 1) ** 2
    # # x = unit_lstsq(A,b)
    # # print(x)
    # # print(np.sum(x ** 2))
    # t = np.linspace(-1000,1000,1000)
    # plt.plot(t, [f(i) for i in t])
    # plt.show()
    # l = unit_lstsq_svd(A,b,1.0)
    # print(l)
    # x = ridge(A,b,l)
    # print(x)
    # print(np.sum(x ** 2))
    x, loss = projective_lstsq(A,b)
    print(x)
    print(loss)

