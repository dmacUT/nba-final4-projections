import pandas as pd
import numpy as np 
import random

def bootstrap(arr, iterations=10):
    """Create a series of bootstrapped samples of an input array.
    Parameters
    ----------
    arr: Numpy array
        1-d numeric data
    iterations: int, optional (default=10000)
        Number of bootstrapped samples to create.
    Returns
    -------
    boot_samples: list of arrays
        A list of length iterations, each element is array of size of input arr
    """
    cols = list(arr.columns.values)
    
    if type(arr) != np.ndarray:
        arr = np.array(arr)

    if len(arr.shape) < 2:
        arr = arr[:, np.newaxis]
        # [:, np.newaxis] increases the dimension of arr from 1 to 2

    nrows = arr.shape[0]
    boot_samples = []
    df_list = []
    
    for _ in range(iterations):
        row_inds = np.random.randint(nrows, size=nrows)
        # because of the [:, np.newaxis] above 
        # the following will is a 1-d numeric data with the same size as the input arr
        boot_sample = arr[row_inds, :]
        
        #Rejoin positioned dataframes
        dfs = pd.DataFrame(boot_sample, columns=cols)
        df_list.append(dfs)
        
    df_merged = pd.concat(df_list)
    colsnoTM_x = cols
    colsnoTM_x.pop(1)
    df_merged[colsnoTM_x] = df_merged[colsnoTM_x].apply(pd.to_numeric)

    return df_merged

def split_1(df, test_size=.3):
    '''
    parameters: dataframe to split
    returns: 
    '''
    #randlist has a random index for % length of index
    test_randlist = random.sample(range(len(df.index.values)), int((len(df.index.values)*test_size)))
    final_test_df = df.iloc[test_randlist]
    final_test_X = final_test_df.drop('HomeCourt_x', axis=1)
    final_test_y = final_test_df['HomeCourt_x']
    train_list = [i for i in list(df.index.values) if i not in test_randlist]
    print(np.min(train_list), np.max(train_list))
    first_train_df = df.iloc[train_list]
    return first_train_df, final_test_X, final_test_y
