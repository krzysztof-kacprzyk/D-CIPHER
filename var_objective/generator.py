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
    

def generate_fields(pdes, conditions, grid, noise_ratio, seed=0):
    
    np.random.seed(seed)

    sample_list = []

    num_samples = conditions.get_num_samples()

    for i in range(num_samples):
        
        observed_scalar_field_list = []

        field_functions = pdes.get_solution(conditions.get_condition_functions(i))

        for field_function in field_functions:
            
            raw_scalar_field = field_function(grid)

            signal = np.std(raw_scalar_field)

            observed_scalar_field = raw_scalar_field + np.random.normal(0,noise_ratio*signal,size=grid.shape)

            observed_scalar_field_list.append(observed_scalar_field)
        
        sample_list.append(np.stack(observed_scalar_field_list, axis=0))

    samples = np.stack(sample_list, axis=0)

    return samples




