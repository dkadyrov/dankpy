import pandas as pd


def remove_outliers(df: pd.DataFrame, col, iqr_coefficient: float = 2) -> pd.DataFrame:
    """
    Removes outliers from a dataframe column using the IQR method

    Args:
        df (pd.DataFrame): DataFrame to remove outliers from
        col (_type_): Column name to remove outliers from
        iqr_coefficient (float, optional): IQR coefficient. Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    distance = iqr_coefficient * (Q3 - Q1)
    lower_fence, upper_fence = Q1 - distance, Q3 + distance

    # Keep rows where values are within the bounds
    df_out = df.loc[~((df[col] < lower_fence) | (df[col] > upper_fence))]

    return df_out
