import numpy as np
def normalize(arr):
    length = len(arr)
    
    total_sum = 0
    for i in range(0, length):
        total_sum = total_sum + arr[i]
    
    mean = total_sum/length

    return (arr - mean)/(np.max(arr) - np.min(arr))