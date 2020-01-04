from sklearn import preprocessing
from scipy import stats

import pandas as pd
import numpy as np

def drop_outliers(data, zscore):
    """
    Select outliers further away from zscore standard
    deviations from the mean

    -----------
    Args:

    data (DataFrame) - df with feature data

    feature (string) - name of feature column in data

    zscore (int) - limit to cut outliers

    -----------
    Returns:

    df - DataFrame without outliers
    """
    # select outliers
    outliers = []
    for col in data:
        df = data[np.abs(stats.zscore(data[col])) > zscore]
        outliers.extend(df.index.values)

    # drop
    return data.drop(outliers)


def standardize(data):
    """
    Standardize data, so it can be zero-centered and
    represented in standard deviations units

    X = (X - mean(X)) / std(X)

    -----------
    Args:

    data (DataFrame) - df with feature data

    -----------
    Returns:

    df - standardized DataFrame
    """
    data = data.apply(lambda x: x.astype(float))
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(data)
    df = pd.DataFrame(scaled_df, columns=data.columns, index=data.index)
    return df


def normalize(df):
    pass


def select_active_period(data, empty_value, response):
    """
    Select period on which the stock is actively being
    traded on the market

    -----------
    Args:

    data (DataFrame) - df with time series data

    feature (string) - name of column in data

    -----------
    Returns:

    df - DataFrame within active period
    """
    df = data[response].replace(empty_value, np.nan)
    df = df.astype(float)
    df = df.interpolate(limit_area='inside')
    return data[~np.isnan(df)]


def drop_empty_features(data):
    """
    Drop columns of data whose values are all empty

    -----------
    Args:

    data (DataFrame) - df with feature values

    -----------
    Returns:

    df - DataFrame without empty columns
    """
    for col in data.columns:
        if (data[col] == '-').all():
            data.drop(col, axis=1, inplace=True)

    return data


def time_diff(data):
    """
    Take time difference between features of assets

    -----------
    Args:

    data (DataFrame) - df with time series data

    -----------
    Returns:

    df - DataFrame with time difference values
    """
    # take time difference
    df = data[1:] - data.shift(1)[1:]
    df = df.replace(float('nan'), 0)
    return df


def ratio(nume, deno):
    """
    Take ratio of two df

    -----------
    Args:

    end_df (DataFrame) - df with later values

    ini_df (DataFrame) - df with earlier values

    -----------
    Returns:

    df - DataFrame with time difference values
    """
    # take ratio between features
    df = nume / deno
    df = df.replace(float('nan'), 0).replace(float('inf'), 1).replace(float('-inf'), -1)

    return df
