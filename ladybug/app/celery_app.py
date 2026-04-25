from celery import Celery
from ladybug.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ladybug",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "ladybug.tasks.training_tasks",
    ]
)

celery_app.conf.update(
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_TASK_SERIALIZER,
    accept_content=[settings.CELERY_TASK_SERIALIZER],
    task_acks_late=settings.CELERY_TASK_ACKS_LATE,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_ignore_result=False,
    result_expires=3600,

    task_routes={
        "ladybug.tasks.training_tasks.task_data_ingestion": {
            "queue": "cpu"
        },
        "ladybug.tasks.training_tasks.task_data_clean": {
            "queue": "cpu"
        },
        "ladybug.tasks.training_tasks.task_data_labeling": {
            "queue": "cpu"
        },
        "ladybug.tasks.training_tasks.task_data_transform": {
            "queue": "cpu"
        },
        "ladybug.tasks.training_tasks.task_prepare_training_assets": {
            "queue": "cpu"
        },
        "ladybug.tasks.training_tasks.task_run_model_training": {
            "queue": "gpu"
        },
    },
)


celery_app.conf.task_default_queue = "default"

# Set up Celery logging
celery_app.conf.update(
    worker_hijack_root_logger=False,  # <-- don't let Celery hijack root logger
)