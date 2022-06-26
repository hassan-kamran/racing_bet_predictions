import pandas as pd

def one_hot_encoding(df, columns):
    df = df.copy()
    for col, pref in columns.items():
        dummy = pd.get_dummies(df[col], prefix=pref)
        dummy.astype(bool)
        df = pd.concat([df, dummy], axis=1)
        df = df.drop(col, axis=1)
    return df

def drop_sparse(df, threshold):
    for col in df.columns:
        if df[col].count() < threshold:
            df.drop(col, axis=1, inplace=True)
    return None