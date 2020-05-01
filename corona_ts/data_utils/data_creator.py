import pandas as pd
from typing import List 

def loop_through_locations(df:pd.DataFrame, column:str='full_county', cut_threshold=60)->List:
    """
    Function to split data-frame based on state or county.
    Returns a list of dataframes.
    """
    df_county_list = []
    df['full_county'] = df['state'] + "_" + df['county'] 
    for code in df[column].unique():
        mask = df[column] == code
        df_code = df[mask]
        ts_count = len(df_code)
        if ts_count > cut_threshold:
            df_county_list.append(df_code)
    return df_county_list 

def incorporate_data(df):
    pass 
    