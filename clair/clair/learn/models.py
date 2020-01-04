import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class ols:
    """
    Create an Ordinary Least Squares model

    -----------
    Args:

    features (DataFrame) - features df

    response (string) - response df
    """

    def __init__(self, features, response, activation):
        # create model
        X = features.values.astype(float)
        Y = response.values.astype(float)
        self.model = sm.OLS(Y, X).fit()

    def predict(self, features):
        """
        Overwrite statsmodel predict method for simplification
        purposes
        """
        return self.model.predict(features)


class logit:
    """
    Create a Logistic Regression model

    -----------
    Args:

    features (DataFrame) - features df

    response (string) - response df
    """

    def __init__(self, features, response, activation):
        # create model
        X = features.values.astype(float)
        Y = response.values.astype(float)
        self.model = LogisticRegression(random_state=0).fit(X, Y)

    def predict(self, features):
        """
        Overwrite sklearn predict method for simplification
        purposes
        """
        return self.model.predict(features)


class mlp:
    """
    Create a multi-layer perceptron (MLP) algorithm that
    trains using backpropagation.

    -----------
    Args:

    features (DataFrame) - features df

    response (string) - response df
    """

    def __init__(self, features, response, activation):
        # create model
        X = features.values.astype(float)
        Y = response.values.astype(float)
        model = MLPClassifier(activation=activation, random_state=0)
        self.model = model.fit(X, Y)

    def predict(self, features):
        """
        Overwrite sklearn predict method for simplification
        purposes
        """
        return self.model.predict(features)
