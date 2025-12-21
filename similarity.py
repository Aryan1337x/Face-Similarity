import numpy as np
import math

def calculate_euclidean_distance(vec1, vec2):
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same number of dimensions")

    diff = vec1 - vec2
    squared_diff = np.square(diff)
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    
    return distance

def explain_calculation_step(vec1, vec2, limit=5):
    pass
