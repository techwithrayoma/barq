import mlflow
from mlflow.exceptions import MlflowException
from typing import Optional

class MLflowTracker:
    """
    MLflow tracking service wrapper
    """
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        artifact_root: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name (str): Name of the MLflow experiment
            tracking_uri (str): MLflow tracking server URI
            artifact_root (str, optional): Location to store artifacts (S3, GCS, local, etc.)
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        if artifact_root:
            self.artifact_root = artifact_root
        else:
            self.artifact_root = None

        # Create or get experiment
        try:
            mlflow.set_experiment(experiment_name)
        except MlflowException as e:
            raise RuntimeError(f"Failed to set MLflow experiment: {e}")

    def _ensure_run(self):
        """Guarantee there is an active MLflow run"""
        if mlflow.active_run() is None:
            mlflow.start_run()
            
    def start_run(self, run_name: Optional[str] = None):
        """
        Start an MLflow run
        """
        return mlflow.start_run(run_name=run_name)

    def log_param(self, params: dict):
        """
        Log multiple params
        """
        try:
            self._ensure_run()
            for k, v in params.items():
                mlflow.log_param(k, v)
        except Exception as e:
            raise RuntimeError(f"LadyBug ✿ failed to log params: {e}")
        
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log multiple metrics
        """
        try:
            self._ensure_run()
            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=step)
        except Exception as e:
            raise RuntimeError(f"LadyBug ✿ failed to log metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file or directory as an artifact
        """
        mlflow.log_artifact(local_path, artifact_path)

    def end_run(self, status: str = "FINISHED"):
        """
        End the MLflow run with status
        """
        try:
            mlflow.end_run(status=status)
        except Exception as e:
            raise RuntimeError(f"LadyBug ✿ failed to end MLflow run: {e}")
