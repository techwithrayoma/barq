from fastapi import APIRouter
from ladybug.tasks.training_tasks import task_run_model_training

router = APIRouter(prefix="/training", tags=["training"])

@router.get("/task")
async def send_email():
    task = task_run_model_training.delay(version="v1", model_name="ladybug")
    return {"status": "accepted", "task_id": task.id}

