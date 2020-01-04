# custom imports
from clair.utils.metrics import f1_score

def forward_stepwise(features, response, model_class, activation=None):
    """
    Create a model designed by forward feature selection,
    evaluated by f1_score
    -----------
    Args:

    data (DataFrame) - df with feature columns

    response (DataFrame) - df with feature column

    -----------
    Returns:

    model - an "optimal" fitted model
    """

    # set of features
    selected = []
    remaining = set(features.columns)
    current_model_score, best_score = 0.0, 0.0

    while remaining and current_model_score == best_score:
        # select best feature
        candidate_score = []
        for candidate in remaining:
            # predict
            df = features[selected + [candidate]]
            model = model_class(df, response, activation)
            y_pred = model.predict(df)

            # calculate f1 score
            score = f1_score(response, y_pred)
            candidate_score.append((score, candidate))

        # add feature with best score to model
        candidate_score.sort()
        best_score, best_candidate = candidate_score.pop()
        if current_model_score < best_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_model_score = best_score
        else:
            break

    # return best fitted model
    if len(selected) == 0:
        return None, None, None
    else:
        model = model_class(features[selected], response, activation)
        return model, selected, current_model_score


def best_subset():
    pass