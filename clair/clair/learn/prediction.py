# standard imports
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd

# custom imports
from clair.learn.feature_selection import forward_stepwise
from clair.utils.metrics import f1_score

def time_series(args):
    data, response, model_class, activation = args
    """
    Predict response based on a model created with forward
    stepwise selection
    -----------
    Args:

    data (dictionary) - dict with time series data for every
    asset

    response (string) - name of dependent variable column

    model_class (clair.models) - model to be used

    activation (string) - name of activation function

    -----------
    Returns:

    results - a dict containing several informations about
    model performance and its coefficients
    """

    results = recursive_dd()

    for asset in data:
        # split data
        x = data[asset].drop([response], axis=1)
        y = data[asset][response]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # fit model
        model, param_labels, train_score = forward_stepwise(x_train, y_train, model_class, activation)

        # skip
        if model is None:
            continue

        # predict
        y_pred = model.predict(x_test[param_labels])
        test_score = f1_score(y_test, y_pred)

        # save model results
        results[asset]['model'] = model
        results[asset]['param_labels'] = param_labels

        # save consolidated results
        value = pd.DataFrame([[asset,
                            train_score,
                            test_score]],
                            columns=['asset_name',
                                    'train_score',
                                    'test_score'])
        aux = results.setdefault('consolidated', pd.DataFrame())
        results['consolidated'] = aux.append(value)

    return results


def cross_section(args):
    data, response, model_class, activation = args
    """
    Predict response based on a model created with forward
    stepwise selection
    -----------
    Args:

    data (dictionary) - dict with cross section data

    -----------
    Returns:

    results - a dict containing several informations about
    the model performance and coefficients
    """

    results = recursive_dd()

    last_date = None

    for date in data:
        # ignore first iteration
        if last_date is None:
            last_date = date
            continue

        # ignore empty data
        if (data[last_date]['signal'] != 1).all():
            last_date = date
            continue

        # fit model
        x_train = data[last_date].drop('signal', axis=1)
        y_train = data[last_date]['signal']
        model, param_labels, train_score = forward_stepwise(x_train, y_train, model_class, activation)

        # skip
        if model is None:
            last_date = date
            continue

        # predict
        x_test = data[date].drop('signal', axis=1)
        y_test = data[date]['signal']
        y_pred = model.predict(x_test[param_labels])
        test_score = f1_score(y_test, y_pred)

        # save model results
        results[date]['model'] = model
        results[date]['param_labels'] = param_labels

        # save consolidated results
        value = pd.DataFrame([[date,
                            train_score,
                            test_score]],
                            columns=['date',
                                    'train_score',
                                    'test_score'])
        aux = results.setdefault('consolidated', pd.DataFrame())
        results['consolidated'] = aux.append(value)

        # i++
        last_date = date

    return results


# custom defaultdict
def recursive_dd():
    return defaultdict(recursive_dd)
