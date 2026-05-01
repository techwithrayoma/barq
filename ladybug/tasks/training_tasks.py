import time

from app.celery_app import celery_app
from ladybug.db.database import SessionLocal
from ladybug.core.config import get_settings
from ladybug.enum.pipeline_step import PipelineStep
from ladybug.pipeline.train_pipeline import TrainPipeline
from ladybug.core.logger import pipeline_logger
from core.logger import TaskIdFilter


logger = pipeline_logger

# ---- Attach task filter
task_filter = TaskIdFilter()
pipeline_logger.addFilter(task_filter)


# ================= PIPELINE BUILDER =================
def build_pipeline(version, model_name, task_id):
    settings = get_settings()

    return TrainPipeline(
        config=settings,
        version=version,
        model_name=model_name,
        store_s3=True,
        store_local=True,
        task_id=task_id,
    )


# ================= HELPERS =================
def get_queue(task):
    return task.request.delivery_info.get("routing_key", "unknown")


# ================= INGESTION =================
@celery_app.task(bind=True, queue="cpu", name="ladybug.tasks.training_tasks.task_data_ingestion")
def task_data_ingestion(self, version: str, model_name: str):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    db = None
    start_time = time.perf_counter()

    try:
        queue = get_queue(self)

        logger.info(
            f"[{PipelineStep.INGESTION}] started | "
            f"version={version} | queue={queue}"
        )

        db = SessionLocal()

        pipeline = build_pipeline(version, model_name, task_id)

        step_start = time.perf_counter()
        pipeline.create_folders()
        logger.info(f"[{PipelineStep.STORAGE}] folders ready in {time.perf_counter() - step_start:.2f}s")

        step_start = time.perf_counter()
        pipeline.data_ingestion(db)
        logger.info(f"[{PipelineStep.INGESTION}] done in {time.perf_counter() - step_start:.2f}s")

    except Exception as e:
        logger.error(f"[{PipelineStep.INGESTION}] failed: {e}", exc_info=True)
        raise

    finally:
        if db:
            db.close()


# ================= CLEAN =================
@celery_app.task(bind=True, queue="cpu", name="ladybug.tasks.training_tasks.task_data_clean")
def task_data_clean(self, version: str, model_name: str):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    start_time = time.perf_counter()

    try:
        queue = get_queue(self)

        logger.info(
            f"[{PipelineStep.CLEAN}] started | version={version} | queue={queue}"
        )

        pipeline = build_pipeline(version, model_name, task_id)

        step_start = time.perf_counter()
        pipeline.data_clean()

        logger.info(f"[{PipelineStep.CLEAN}] done in {time.perf_counter() - step_start:.2f}s")

    except Exception as e:
        logger.error(f"[{PipelineStep.CLEAN}] failed: {e}", exc_info=True)
        raise


# ================= LABELING =================
@celery_app.task(bind=True, queue="cpu", name="ladybug.tasks.training_tasks.task_data_labeling")
def task_data_labeling(self, version: str, model_name: str):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    db = None
    start_time = time.perf_counter()

    try:
        queue = get_queue(self)

        logger.info(
            f"[{PipelineStep.LABELING}] started | version={version} | queue={queue}"
        )

        db = SessionLocal()

        pipeline = build_pipeline(version, model_name, task_id)


        step_start = time.perf_counter()
        pipeline.data_labeling(db)

        logger.info(f"[{PipelineStep.LABELING}] done in {time.perf_counter() - step_start:.2f}s")


    except Exception as e:
        logger.error(f"[{PipelineStep.INGESTION}] failed: {e}", exc_info=True)
        raise

    finally:
        if db:
            db.close()


# ================= TRANSFORMATION =================
@celery_app.task(bind=True, queue="cpu", name="ladybug.tasks.training_tasks.task_data_transform")
def task_data_transform(self, version: str, model_name: str):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    try:
        queue = get_queue(self)

        logger.info(
            f"[{PipelineStep.TRANSFORMATION}] started | version={version} | queue={queue}"
        )

        pipeline = build_pipeline(version, model_name, task_id)

        step_start = time.perf_counter()
        pipeline.data_transformation()

        logger.info(f"[{PipelineStep.TRANSFORMATION}] done in {time.perf_counter() - step_start:.2f}s")

    except Exception as e:
        logger.error(f"[{PipelineStep.TRANSFORMATION}] failed: {e}", exc_info=True)
        raise


# ================= TRAINING ASSETS =================
@celery_app.task(bind=True, queue="cpu", name="ladybug.tasks.training_tasks.task_prepare_training_assets")
def task_prepare_training_assets(self, version: str, model_name: str):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    try:
        queue = get_queue(self)

        logger.info(
            f"[{PipelineStep.TRAINING}] started | version={version} | queue={queue}"
        )

        pipeline = build_pipeline(version, model_name, task_id)

        step_start = time.perf_counter()
        pipeline.prepare_training_assets()

        logger.info(f"[{PipelineStep.TRAINING}] done in {time.perf_counter() - step_start:.2f}s")

    except Exception as e:
        logger.error(f"[{PipelineStep.TRAINING}] failed: {e}", exc_info=True)
        raise


# ================= GPU MODEL TRAINING =================
@celery_app.task(bind=True, queue="gpu", name="ladybug.tasks.training_tasks.task_run_model_training")
def task_run_model_training(self, version: str, model_name: str):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    try:
        queue = get_queue(self)

        logger.info(
            f"[{PipelineStep.TRAINING}] GPU TRAINING started | version={version} | queue={queue}"
        )

        pipeline = build_pipeline(version, model_name, task_id)

        step_start = time.perf_counter()
        pipeline.run_model_training()

        logger.info(f"[{PipelineStep.TRAINING}] GPU training done in {time.perf_counter() - step_start:.2f}s")

    except Exception as e:
        logger.error(f"[{PipelineStep.TRAINING}] GPU training failed: {e}", exc_info=True)
        raise