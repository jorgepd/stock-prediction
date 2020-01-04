import pandas as pd
import os

def screening(base_dir):
    """
    Read Economatica's screening data

    -----------
    Args:

    base_dir (string) - directory location of the data

    -----------
    Returns:

    df - a DataFrame containing all data
    """
    df = pd.read_excel(base_dir, header=3)
    return df.drop(labels=df.columns[0], axis='columns')


def matrixx(base_dir, asset_codes):
    """
    Read Economatica's matrixx data

    -----------
    Args:

    base_dir (string) - directory location of the data

    asset_codes (list) - list of asset codes in xlsx files

    -----------
    Returns:

    matrixx - a dictionary whose keys are the feature names,
    and values are DataFrames containing data for all assets
    """
    matrixx = {}
    entries = os.scandir(base_dir)

    for entry in entries:
        if entry.name.endswith('.xlsx'):
            print(entry.name)

            # read and format data
            df = pd.read_excel(base_dir + entry.name, header=3)
            df.columns = ['Date', *asset_codes]

            # drop last quarter
            df = df[:135]

            # set date index
            df.set_index('Date', inplace=True)

            # save data
            matrixx[entry.name.replace('.xlsx', '')] = df.replace('-', 0)

    return matrixx
