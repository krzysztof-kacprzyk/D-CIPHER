import numpy as np

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


def get_conditions_set(name):
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
    
    return conditions
        