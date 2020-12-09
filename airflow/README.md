# Instructions 

on how to launch airflow instance

## prerequisite

- docker installed

- traefik revorse proxy is running (check coronawhy-infrastructure repo for details)

- domain name (f.e. `airflow.coronawhy.org` pointing to IP of your VM)

## Steps

1. put a `cred.json` file from GCP service account to `/dags` folder

2. create an env file `airflow.env` to specify traefikhost host

```
traefikhost=airflow.coronawhy.org
```

3. simply run
```
docker-compose --env-file airflow.env up -d
```

You should be able to access airflow server at `airflow.coronawhy.org` and see DAGs from folder `/dags` in the list 
