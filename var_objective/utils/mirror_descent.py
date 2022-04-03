import numpy as np


class MirrorDescentSimplex:

    def __init__(self,n, gradient_function, initial_guess=None):
        self.n = n
        self.gradient_function = gradient_function

        if initial_guess is None:
            self.x = np.ones(self.n) / self.n
        else:
            self.x = initial_guess

    def optimize(self,num_epochs,lr=0.01,tol=1e-10):
      
        for i in range(1,num_epochs+1):
            g = self.gradient_function(self.x)
            if np.linalg.norm(g,2) < tol:
                break
            a = lr / np.sqrt(i)
            self.step(g,a)
        
        return self.x

    def step(self,g,a):
        y = self.x * np.exp(-a*g)
        self.x = y / np.sum(y)
        

    
        
if __name__ == "__main__":

    np.random.seed(0)

    m = 10000
    n = 5

    A = np.random.uniform(0.0,1.0,(m,n))
    b = np.random.uniform(0.0,1.0,(m))

    def f(x):
        return np.sum(np.power(np.dot(A,x)-b,2))
    
    def df(x):
        return 2*np.dot(A.T,(np.dot(A,x)-b))

    md = MirrorDescentSimplex(n,df)
    x = md.optimize(1000,0.0001)
    print(x)
    print(f(x))


