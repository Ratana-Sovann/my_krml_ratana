
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