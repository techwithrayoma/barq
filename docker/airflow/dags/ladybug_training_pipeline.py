from airflow import DAG
from airflow.providers.celery.operators.celery import CeleryOperator
from airflow.utils.dates import days_ago

default_args = {"owner": "airflow"}

with DAG(
    dag_id="ladybug_training_pipeline",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
) as dag:

    trigger_ingestion = CeleryOperator(
        task_id="trigger_ingestion",
        task_name="ladybug.app.celery_app.train_tasks.task_data_ingestion",
        queue="cpu",
        broker="amqp://ladybug_user:ladybug_rabbitmq_2222@rabbitmq:5672/ladybug_vhost",
        backend="redis://:ladybug_redis_2222@redis:6379/0",
        args=["v5", "ladybug-model"],
    )
