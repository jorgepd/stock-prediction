import matplotlib.pyplot as plt

def features(plt, data, response, model_class):
    """
    Create a grafic with feature values and a line
    represented by a fitted model_class

    -----------
    Args:

    data (DataFrame) - df with feature and response data

    response (string) - name of response column in data

    model_class (clair model class) - model class which is
    going to be used in the plot

    -----------
    Returns:

    plt - a plot ready to be shown
    """
    # split dependent and independent variables
    feature = (set(data.columns) - set([response])).pop()
    x = data[feature]
    y = data[response]

    # create model
    x_line = x
    model = model_class(x, y)
    y_line = model.predict(x)

    # draw feature values as points
    plt.plot(x, y,'ob', markersize=3)

    # draw model values as line
    plt.plot(x_line, y_line,'-r')

    # return plot
    plt.ylabel(response)
    plt.xlabel(feature)
    return plt


def model_forecast(results, data, response):
    # setup
    x = data.drop(response, axis=1)
    y = data[response]

    # draw dependent value line
    plt.plot(y, '-', label='real', color='blue')

    # draw model values line
    model = results['model']
    param_labels = results['param_labels']
    y_pred = model.predict(x[param_labels])
    plt.plot(y_pred, '--', label='pred', color='red')

    # return plot
    plt.legend()
    return plt