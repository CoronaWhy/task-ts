import pandas as pd


def loop_through_locations(
        df:pd.DataFrame,
        columns_to_consider_for_uniqueness=['country', 'region', 'sub_region'],
        unique_column_name='full_county',
        minimum_datapoints_threshold=60):
    """
    Function to split data-frame based on state or county.
    """
    unique_df_list = []
    for col in columns_to_consider_for_uniqueness:
        df[col] = df[col].fillna('').apply(lambda x: x.replace(" ", "_"))
    df[unique_column_name] = df[columns_to_consider_for_uniqueness[0]].str.cat(df[columns_to_consider_for_uniqueness[1:]], sep="__")
    for code in df[unique_column_name].unique():
        mask = (df[unique_column_name] == code)
        df_code = df[mask].copy()
        ts_count = len(df_code)
        if ts_count > minimum_datapoints_threshold:
            unique_df_list.append(df_code)
    return unique_df_list

def region_df_format(df, sub_region):
  region_df = df[df['sub_region']==sub_region]
  region_df = region_df.sort_values(by='date')
  region_df.index = region_df.date
  region_df['new_cases'] = region_df['cases'].diff()
  region_df['weekday'] = region_df['date'].map(lambda x: x.strftime('%w'))
  return region_df
