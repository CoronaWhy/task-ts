import pandas as pd

# def loop_through_locations(df:pd.DataFrame, column:str='full_county', cut_threshold=60):
#     """
#     Function to split data-frame based on state or county.
#     """
#     df_county_list = []
#     df['full_county'] = df['state'] + "_" + df['county']
#     for code in df['full_county'].unique():
#         mask = df['full_county'] == code
#         df_code = df[mask]
#         ts_count = len(df_code)
#         if ts_count > cut_threshold:
#             df_county_list.append(df_code)
#     return df_county_list


def loop_through_locations(
        df:pd.DataFrame,
        columns_to_consider_for_uniqueness=['country', 'region', 'sub_region'],
        minimum_datapoints_threshold=60):
    """
    Function to split data-frame based on state or county.
    """
    unique_df_list = []
    for col in columns_to_consider_for_uniqueness:
        df[col] = df[col].fillna('').apply(lambda x: x.replace(" ", "_"))
    df['region_identifier'] = df[columns_to_consider_for_uniqueness[0]].str.cat(df[columns_to_consider_for_uniqueness[1:]], sep="__")
    for code in df['region_identifier'].unique():
        mask = (df['region_identifier'] == code)
        df_code = df[mask].copy()
        ts_count = len(df_code)
        if ts_count > minimum_datapoints_threshold:
            unique_df_list.append(df_code)
    return unique_df_list

