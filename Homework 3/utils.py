import pandas as pd

def is_k_anonym(df:pd.DataFrame, feature_columns, k):
    gbdc = df[feature_columns].groupby(feature_columns) # Group by feature columns
    _is_k_anonym = (gbdc[feature_columns[0]].count().values >= k).all()
    return _is_k_anonym

def is_l_diverse(df:pd.DataFrame, sensitive_column:str, feature_columns:list, l:int=2):
    gbdc = df[[sensitive_column]+feature_columns].groupby(feature_columns) # Group by feature columns
    _is_l_diverse = (gbdc[sensitive_column].nunique() >= l).all()
    return _is_l_diverse

