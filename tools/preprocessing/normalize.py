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
        # for row in range(0, row_len):
        #     total_sum = total_sum + arr[row, col]
        total_sum = arr[:, col].sum(axis=0)

        mean = total_sum/row_len
        max_min = (np.max(arr[:, col]) - np.min(arr[:, col]))
        if max_min == 0:
            res[:, col] = 0
        else:
            res[:, col] = np.divide((arr[:, col] - mean), max_min)

    return res