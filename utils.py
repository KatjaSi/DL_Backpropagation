"""
Helper functions here
"""
import random
import numpy as np

def init_weights(shape, range_min, range_max): # ex shape = (2,3) means we need to generate matrix with 2 rows and 3 columns,
    return np.random.uniform(range_min, range_max, size = shape)

def sample_indices(arr, n_elements):
    """
    Returns indies of n_elements random elements from array arr
    """
    indices = random.sample(range(0, len(arr)), n_elements)
    return indices

def str_to_bool(str):
    str2 = str.strip()
    return str2=='True' or str2=='true'

def main():
    print(init_weights((2), -0.1, 0.1))
    

if __name__ == '__main__':
    main()