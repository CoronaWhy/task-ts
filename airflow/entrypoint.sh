#!/bin/sh
export AIRFLOW_HOME=/work
airflow initdb
airflow webserver &
airflow scheduler
