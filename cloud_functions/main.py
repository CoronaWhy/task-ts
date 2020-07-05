from google.cloud import storage
import pandas as pd
import csv
import io

def loop_location_to_csv(data, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This generic function logs relevant data when a file is changed.

    Args:
        data (dict): The Cloud Functions event payload.
        context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
        None; the output is written to Stackdriver Logging
    """

    print('Event ID: {}'.format(context.event_id))
    print('Event type: {}'.format(context.event_type))
    
    input_bucket_name = data['bucket']
    source_file = data['name']
    print('File: {}'.format(source_file))
    uri = 'gs://{}/{}'.format(input_bucket_name, source_file)
    df = pd.read_csv(uri)
    columns_to_consider_for_uniqueness=['country', 'region', 'sub_region']
    unique_column_name='full_county'
    minimum_datapoints_threshold=60
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
            df_code.reset_index(drop = True).loc[:, ~df.columns.str.contains('^Unnamed')].to_csv('gs://{}/{}/{}.csv'.format(input_bucket_name, source_file[:source_file.rindex('/')],code, index = False))
