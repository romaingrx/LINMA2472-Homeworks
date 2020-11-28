import pandas as pd

def is_k_anonym(df:pd.DataFrame, feature_columns:str, k:int=2):
    """Determine if the dataset respect the k-anonymity
       source : https://en.wikipedia.org/wiki/K-anonymity

    Args:
        df (pd.DataFrame): the dataset 
        feature_columns (list): the feature/quasi-identifiers columns ot look for the k-anonymity
        k (int, optional): The corresponding k to respect. Defaults to 2.

    Returns:
        [bool]: True if the count of all set of quasi-identifiers are greater or equal to k
    """

    gbdc = df[feature_columns].groupby(feature_columns) # Group by feature columns
    _is_k_anonym = (gbdc[feature_columns[0]].count().values >= k).all()
    return _is_k_anonym

def is_l_diverse(df:pd.DataFrame, sensitive_column:str, feature_columns:list, l:int=2) -> bool:
    """determine if the dataset respects the l-diversity
       source : https://en.wikipedia.org/wiki/L-diversity

    Args:
        df (pd.DataFrame): the dataset 
        sensitive_column (str): the sensitive column to preserve from anonymity
        feature_columns (list): the feature/quasi-identifiers columns ot look for the l-diversity
        l (int, optional): The corresponding l to respect. Defaults to 2.

    Returns:
        bool: True if the number of unique sensitive informations of all set of quasi-identifiers are greater or equal to l
    """
    gbdc = df[[sensitive_column]+feature_columns].groupby(feature_columns) # Group by feature columns
    _is_l_diverse = (gbdc[sensitive_column].nunique() >= l).all()
    return _is_l_diverse

def get_sensitive_frequences(df:pd.DataFrame, sensitive_column:str) -> dict:
    """get the frequencies for each sensitive value

    Returns:
        dict: a dictionnary with the coreesponding frequency for each sensitive value
    """
    total_count = len(df)
    sensitive_freqs = dict()
    counts = df.groupby(sensitive_column)[sensitive_column].agg('count')
    for value, count in counts.to_dict().items():
        sensitive_freqs[value] = count / total_count
    return sensitive_freqs

def t_closeness(df:pd.DataFrame, sensitive_column:str, feature_columns:list):
    """return the maximum distance between the distribution of a sensitive value in a \
    set of quasi-identifiers and the distribution of this same sensitive value in the \
    all database

    Args:
        df (pd.DataFrame): the dataset 
        sensitive_column (str): the sensitive column to preserve from anonymity
        feature_columns (list): the feature/quasi-identifiers columns ot look for the l-diversity

    Returns:
        int: the maximum distance
    """
    # First get all frequences for each unique sensitive value
    sensitive_freqs = get_sensitive_frequences(df, sensitive_column) 
    d_max = 0.
    gb = df[[sensitive_column]+feature_columns].groupby(feature_columns)[sensitive_column]

    for idx, (name, group) in enumerate(gb):
        sensitive_values = group.values
        for value in group.unique():
            p = sum(sensitive_values == value) / len(group)
            distance = abs(sensitive_freqs[value] - p)
            d_max = max(d_max, distance)
    return d_max