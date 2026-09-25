import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    l = len(data)
    sorted_data = sorted(data)

    # mean
    mean = np.mean(data)

    # median
    if l % 2 == 0:
        median = (sorted_data[l//2 -1] + sorted_data[l//2]) / 2
    else:
        median = sorted_data[l//2]
    
    # mode
    count = {}
    max_count = 0
    mode = None
    for val in sorted_data:
        count[val] = count.get(val, 0) + 1
        
        if count[val] > max_count:
            max_count = count[val]
            mode = val

    # variance
    var = sum((x - mean)**2 for x in data) / l

    # std
    std = var ** 0.5

    # percentiles
    q25, q50, q75 = np.percentile(data, [25, 50, 75])

    # iq range
    iq_range = q75 - q25

    return {
        'mean':mean,
        'median':median,
        'mode':mode,
        'variance':var,
        'standard_deviation':std,
        '25th_percentile':q25,
        '50th_percentile':q50,
        '75th_percentile':q75,
        'interquartile_range':iq_range
    }