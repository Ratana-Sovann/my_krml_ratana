
def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {rmse(y_actuals, y_preds)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")


def print_classifier_auc(y_preds, y_actuals, set_name=None):
    """Print the AUC score for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted probabilities or scores for the positive class
    y_actuals : Numpy Array
        Actual binary target values (0 or 1)
    set_name : str, optional
        Name of the set to be printed

    Returns
    -------
    None
    """
    from sklearn.metrics import roc_auc_score

    auc_score = roc_auc_score(y_actuals, y_preds)
    print(f"AUC {set_name}: {auc_score:.4f}")

def assess_regressor_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its RMSE and MAE scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_regressor_scores(y_preds=preds, y_actuals=target, set_name=set_name)

def fit_assess_regressor(model, X_train, y_train, X_val, y_val):
    """Train a regressor model, print its RMSE and MAE scores on the training and validation set and return the trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_val : Numpy Array
        Features for the validation set
    y_val : Numpy Array
        Target for the validation set

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train)
    assess_regressor_set(model, X_train, y_train, set_name='Training')
    assess_regressor_set(model, X_val, y_val, set_name='Validation')
    return model