import pandas as pd

def matrixx_to_asset_key(matrixx, asset_codes):
    """
    Reorder matrixx data by asset

    -----------
    Args:

    matrixx (dictionary) - dict whose keys are asset
    features and values are DataFrames containing
    assets values

    asset_codes (iterable) - list of assets to use when
    pivoting dict

    -----------
    Returns:

    asset_dict - reordered dict with assets as keys
    """
    asset_dict = {}

    for asset in asset_codes:
        asset_dict[asset] = pd.DataFrame()

        # pivot asset parameters
        for key in matrixx.keys():
            asset_dict[asset][key] = matrixx[key][asset]

    return asset_dict


def matrixx_to_date_key(matrixx, date_list):
    """
    Reorder matrixx data by date

    -----------
    Args:

    matrixx (dictionary) - dict whose keys are asset
    features and values are DataFrames containing
    assets values

    date_list (iterable) - list of dates to use when
    pivoting dict

    -----------
    Returns:

    date_dict - reordered dict with assets as dates
    """
    date_dict = {}

    for date in date_list:
        date_dict[date] = pd.DataFrame()

        # pivot features
        for key in matrixx.keys():
            date_dict[date][key] = matrixx[key].loc[date]

    return date_dict
