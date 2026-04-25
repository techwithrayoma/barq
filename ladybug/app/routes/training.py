from fastapi import APIRouter
from ladybug.tasks.training_tasks import task_data_ingestion

router = APIRouter(prefix="/training", tags=["training"])

@router.get("/task")
async def send_email():
    task = task_data_ingestion.delay(version="v8", model_name="ladybug")
    return {"status": "accepted", "task_id": task.id}
