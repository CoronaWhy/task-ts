import pandas as pd

def loop_through_locations(df:pd.DataFrame, column:str='full_county', cut_threshold=60):
    """
    Function to split data-frame based on state or county.
    """
    df_county_list = []
    df['full_county'] = df['state'] + "_" + df['county'] 
    for code in df['full_county'].unique():
        mask = df['full_county'] == code
        df_code = df[mask]
        ts_count = len(df_code)
        if ts_count > cut_threshold:
            df_county_list.append(df_code)
    return df_county_list 

