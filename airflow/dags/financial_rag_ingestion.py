from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from vectorstore.build_index import main as build_index_main

default_args = {
    "owner": "abhishek",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="financial_rag_ingestion",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    rebuild_index = PythonOperator(
        task_id="rebuild_index",
        python_callable=build_index_main,
    )
