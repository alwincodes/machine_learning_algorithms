import numpy as np
def normalize(arr):
    length = len(arr)
    
    total_sum = 0
    for i in range(0, length):
        total_sum = total_sum + arr[i]
    
    mean = total_sum/length

    return (arr - mean)/(np.max(arr) - np.min(arr))


def multi_col_normalize(arr):
    columns = arr.shape[1]
    res = np.zeros(arr.shape)
    #loop for each columns
    for col in range(0, columns):
        #get the length of the row
        row_len = len(arr[:, col])
        total_sum = 0
        for row in range(0, row_len):
            total_sum = total_sum + arr[row, col]

        mean = total_sum/row_len
        res[:, col] = (arr[:, col] - mean)/(np.max(arr[:, col]) - np.min(arr[:, col]))

    return res