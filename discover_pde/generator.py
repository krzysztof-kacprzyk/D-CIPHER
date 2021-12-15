import numpy as np


def generate_fields(functions, grid, num_samples, noise_ratio, seed=0):
    
    np.random.seed(seed)

    sample_list = []

    for i in range(num_samples):
        
        observed_scalar_field_list = []

        for f in functions:
            raw_scalar_field = f(grid.by_axis())

            signal = np.std(raw_scalar_field)

            observed_scalar_field = raw_scalar_field + np.random.normal(0,noise_ratio*signal,size=grid.shape)

            observed_scalar_field_list.append(observed_scalar_field)
        
        sample_list.append(np.stack(observed_scalar_field_list, axis=0))

    samples = np.stack(sample_list, axis=0)

    return samples




