from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Connection
from airflow import DAG, settings
# Connects to GoogleCloud
from airflow.contrib.hooks.gcs_hook import GoogleCloudStorageHook
# These args will get passed on to each operator
import json, os
import pandas as pd
import numpy as np
# Needed for load_data
from corona_ts.data_utils.data_crawler import load_data

# Setting environment key for GCP path
os.environ["GCP_KEY_PATH"] = '/home/efawe/airflow/dags/task-ts-53924e1e3506.json'

def add_gcp_connection(**kwargs):
    new_conn = Connection(
            conn_id="google_cloud_default",
            conn_type='google_cloud_platform',
    )
    extra_field = {
        "extra__google_cloud_platform__scope": "https://www.googleapis.com/auth/cloud-platform",
        "extra__google_cloud_platform__project": "task-ts",
        "extra__google_cloud_platform__key_path": os.environ["GCP_KEY_PATH"]
    }

    session = settings.Session()

    #checking if connection exist
    if session.query(Connection).filter(Connection.conn_id == new_conn.conn_id).first():
        my_connection = session.query(Connection).filter(Connection.conn_id == new_conn.conn_id).one()
        my_connection.set_extra(json.dumps(extra_field))
        session.add(my_connection)
        session.commit()
    else: #if it doesn't exit create one
        new_conn.set_extra(json.dumps(extra_field))
        session.add(new_conn)
        session.commit()

def data_to_GCS(csv_name: str, folder_name: str,
                   bucket_name="task_ts_data", **kwargs):
    hook = GoogleCloudStorageHook()
    data = load_data()
    df = pd.DataFrame(data=data)
    df.to_csv('corona_data.csv', index=False)
    columns_to_consider_for_uniqueness=['country', 'region', 'sub_region']
    unique_column_name='full_county'
    minimum_datapoints_threshold=60
    """
    Function to split data-frame based on state or county.
    """
    unique_df_list = []
    for col in columns_to_consider_for_uniqueness:
        df[col] = df[col].fillna('').apply(lambda x: x.replace(" ", "_"))
    df[unique_column_name] = df[columns_to_consider_for_uniqueness[0]].str.cat(df[columns_to_consider_for_uniqueness[1:]], sep="__"))
    for i, g in df.groupby('full_county'):
        df_code = g.copy()
        ts_count = len(df_code)
        if ts_count > minimum_datapoints_threshold:
            df_code.reset_index(drop = True).loc[:, ~df.columns.str.contains('^Unnamed')].to_csv('{}.csv'.format(i), index = False)
            hook.upload(bucket_name,
                    object='{}/{}.csv'.format(folder_name, i),
                    filename='{}.csv'.format(i),
                    mime_type='text/csv')
    """ Function for full data pull
    df.to_csv('corona_data.csv', index=False)
    hook.upload(bucket_name,
                object='{}/{}.csv'.format(folder_name, csv_name),
                filename='corona_data.csv',
                mime_type='text/csv')
    """



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
dag = DAG(
    'load_data_forecast',
    default_args=default_args,
    description='DAG to populate mobility data for forecast team',
    schedule_interval='@daily',
)

activate_GCP = PythonOperator(
        task_id='add_gcp_connection_python',
        python_callable=add_gcp_connection,
        provide_context=True,
        dag = dag,
    )

data_to_GCS_task = PythonOperator(
        task_id='data_to_GCS_python',
        python_callable=data_to_GCS,
        provide_context=True,
        op_kwargs={'csv_name': 'corona_data', 'folder_name': str(datetime.datetime.today().date())},
        dag =dag
    )

dag.doc_md = __doc__

activate_GCP >>  data_to_GCS_task