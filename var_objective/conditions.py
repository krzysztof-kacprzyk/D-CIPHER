import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

import matplotlib.pyplot as plt

from var_objective.grids import EquiPartGrid

class Conditions:
     
    def __init__(self, num_conditions_per_sample):
        self.num_condtions_per_sample = num_conditions_per_sample
        self.conditions = []

    def add_sample(self, functions):
        if len(functions) != self.num_condtions_per_sample:
            raise ValueError("Incorrect number of condition functions")
        else:
            self.conditions.append(functions)

    def get_num_samples(self):
        return len(self.conditions)
    
    def get_condition_functions(self, index):
        return self.conditions[index]

class RandomConditions:

    def __init__(self, num_conditions_per_sample, num_samples, length_scale, mean_range, std_range, seed=0):

        self.num_conditions_per_sample = num_conditions_per_sample
        self.num_samples = num_samples
        self.length_scale = length_scale
        self.mean_range = mean_range
        self.std_range = std_range

        np.random.seed(seed)
        self.seeds = np.random.randint(0,1000000,size=(self.num_samples,self.num_conditions_per_sample))

    def get_num_samples(self):
        return self.num_samples

    def get_condition_functions(self, index):
        return [generate_random_function(self.length_scale, self.mean_range,self.std_range,seed=seed) for seed in self.seeds[index]]

class RandomSources2D:

    def __init__(self, num_sources_per_sample, num_samples, seed=0):
        
        self.num_conditions_per_sample = 2 # positions and charges
        self.num_sources_per_sample = num_sources_per_sample
        self.num_samples = num_samples

        np.random.seed(seed)
        self.seeds = np.random.randint(0,1000000,size=self.num_samples)
    
    def get_num_samples(self):
        return self.num_samples

    def get_condition_functions(self, index):
        return generate_random_sources_2D(self.num_sources_per_sample,seed=self.seeds[index]) 

class RandomSources3D:

    def __init__(self, num_sources_per_sample, num_samples, seed=0):
        
        self.num_conditions_per_sample = 2 # positions and charges
        self.num_sources_per_sample = num_sources_per_sample
        self.num_samples = num_samples

        np.random.seed(seed)
        self.seeds = np.random.randint(0,1000000,size=self.num_samples)
    
    def get_num_samples(self):
        return self.num_samples

    def get_condition_functions(self, index):
        return generate_random_sources_3D(self.num_sources_per_sample,seed=self.seeds[index]) 

class RandomNumbers:

    def __init__(self,num_conditions_per_sample,min_value,max_value,num_samples, seed=0):

        self.num_conditions_per_sample = num_conditions_per_sample
        self.num_samples = num_samples
        self.min_value = min_value
        self.max_value = max_value

        np.random.seed(seed)
        self.seeds = np.random.randint(0,1000000,size=self.num_samples)

    def get_num_samples(self):
        return self.num_samples
    
    def get_condition_functions(self, index):
        return generate_random_number(self.num_conditions_per_sample,self.min_value,self.max_value,seed=self.seeds[index])


def get_conditions_set(name, params={'seed':0, 'num_samples':1}):
    if name == '1Sin':
        conditions = Conditions(1)
        conditions.add_sample([lambda t: np.sin(t)])
    elif name == '1SinSquare':
        conditions = Conditions(1)
        conditions.add_sample([lambda t: np.sin(t)])
        conditions.add_sample([lambda t: np.power(t,2.0)])
    elif name == 'SLM1':
        conditions = Conditions(2)
        conditions.add_sample([lambda x: 5*(-np.power(2*x-1,2)+1), lambda x: np.power(x-1,2)])
    elif name == 'HeatZero':
        conditions = Conditions(1)
        conditions.add_sample([lambda x: np.zeros_like(x)])
    elif name == 'Heat1':
        conditions = Conditions(1)
        conditions.add_sample([lambda x: np.zeros_like(x)])
        conditions.add_sample([lambda x: np.ones_like(x)*5])
        conditions.add_sample([lambda x: np.ones_like(x)*20.0])
        conditions.add_sample([lambda x: x])
        conditions.add_sample([lambda x: 5*x - 10])
        conditions.add_sample([lambda x: 5.0 - x])
        conditions.add_sample([lambda x: 10*np.cos(5*x)])
        conditions.add_sample([lambda x: (x - 1) ** 2])
        conditions.add_sample([lambda x: 2*x])
        conditions.add_sample([lambda x: 2*x + 3*np.cos(x)])
    elif name == 'HeatTuning':
        conditions = Conditions(1)
        conditions.add_sample([lambda x: np.zeros_like(x)])
        conditions.add_sample([lambda x: x])
        conditions.add_sample([lambda x: np.cos(x)])
    elif name == 'HeatRandom':
        length_scale = 0.4
        mean_range = (-10,10)
        std_range = (0.5, 4)
        conditions = RandomConditions(1,params['num_samples'],length_scale, mean_range, std_range, seed=params['seed'])
    elif name == 'TestRandom':
        length_scale = 0.4
        mean_range = (-1,1)
        std_range = (0.5, 2)
        conditions = RandomConditions(1,params['num_samples'],length_scale, mean_range, std_range, seed=params['seed'])
    elif name == 'SourcesRandom2D':
        conditions = RandomSources2D(2,params['num_samples'],seed=params['seed'])
    elif name == 'SourcesRandom3D':
        conditions = RandomSources3D(2,params['num_samples'],seed=params['seed'])
    elif name == 'NumbersRandom1':
        conditions = RandomNumbers(1,1.0,2.0,params['num_samples'],seed=params['seed'])
    elif name == 'NumbersRandom2':
        conditions = RandomNumbers(2,-2.0,2.0,params['num_samples'],seed=params['seed'])
    return conditions

def generate_random_number(num_numbers,min_number,max_number,seed=0):
    np.random.seed(seed)
    return np.random.rand(num_numbers,) * (max_number - min_number) + min_number

def generate_random_sources_2D(num_sources,seed=0):
    np.random.seed(seed)
    locations = []
    for i in range(num_sources):

        while True:
            x0 = np.random.uniform(-1,2)
            x1 = np.random.uniform(-1,2)
            if (0 <= x0 <= 1) and (0 <= x1 <= 1):
                continue
            else:
                locations.append([x0,x1])
                break
    locs = np.stack(locations, axis=0)
    charges = np.random.uniform(-2,2,num_sources)
    return [locs,charges]

def generate_random_sources_3D(num_sources,seed=0):
    np.random.seed(seed)
    locations = []
    for i in range(num_sources):

        while True:
            x0 = np.random.uniform(-1,2)
            x1 = np.random.uniform(-1,2)
            x2 = np.random.uniform(-1,2)
            if (0 <= x0 <= 1) and (0 <= x1 <= 1) and (0 <= x2 <= 1):
                continue
            else:
                locations.append([x0,x1,x2])
                break
    locs = np.stack(locations, axis=0)
    print(locs)
    charges = np.random.uniform(-2,2,num_sources)
    return [locs,charges]


def generate_random_function(length_scale, mean_range, std_range, seed=0):

    np.random.seed(seed)

    mean = np.random.uniform(mean_range[0], mean_range[1])
    std = np.random.uniform(std_range[0], std_range[1])

    gp = GaussianProcessRegressor(kernel=RBF(length_scale=length_scale),random_state=seed)

    def f(x):
        org_shape = x.shape
        covariates = x.reshape(-1,1)
        return gp.sample_y(covariates,random_state=seed).reshape(*org_shape) * std + mean

    return f

if __name__ == '__main__':

    t = np.linspace(0,2,200)

    f = generate_random_function(0.4,mean_range=(-10,10),std_range=(1,2),seed=0)

    plt.plot(t, f(t))
    plt.show()