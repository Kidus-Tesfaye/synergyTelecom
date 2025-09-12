import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def get_mi_score(X: pd.DataFrame, y: pd.Series, discrete_features, random_state = None) -> pd.Series:
    """Returns the mutual information of columns unit scikit-learn's mutual_info_regression function. The array returned by mutual_info_regression is converted
    to a series before it is sorted descendingly. 
    For more info read https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html"""
    
    mi_score = mutual_info_regression(X, y, discrete_features= discrete_features, random_state= random_state)
    mi_score = pd.Series(mi_score, name = "MI score", index = X.columns)
    mi_score.sort_values(ascending= False)

    return mi_score
