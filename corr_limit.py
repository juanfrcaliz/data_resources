import pandas as pd
import numpy as np
import tqdm


def corr_limit(data: pd.DataFrame, var: str, target: str, start=0, end=None, bins=100):
    """
    This function creates a discretization of a variable at different thresholds and calculates the correlation with a 
    target variable for each of the threshold values.
    :param data: pd.DataFrame containing both the variable and the target.
    :param var: Name of the variable.
    :param target: Name of the target.
    :param start: Starting threshold value
    :param end: End threshold value. If not specified, the maximum value of the variable will be taken.
    :param bins: Number of bins in which the values of the variable will be divided.
    :return: pd.DataFrame containing the correlation values for each threshold.
    """
    var_corr = data[[var, target]].copy()
    var_corr.drop(target, axis=1, inplace=True)

    if end is None:
        end = int(data[var].max())
    else:
        end = int(end)

    step = int((end - start) / bins)
    limits = [limit for limit in range(start, end, step)]

    print('Computing correlations for each step:')
    for lim in tqdm.tqdm(limits):
        var_corr[lim] = np.where(data[var] < lim, 1, 0)
        var_corr = var_corr.copy()

    corr = var_corr.drop(var, axis=1).corrwith(data[target])

    min_corr = corr[corr.values == corr.values.min()]
    max_corr = corr[corr.values == corr.values.max()]
    print('Min correlation:')
    display(min_corr)
    print('Max correlation:')
    display(max_corr)

    return corr
