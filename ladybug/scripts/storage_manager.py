from pathlib import Path
import pandas as pd


from ladybug.enum.pipeline_step import PipelineStep, StorageType
from ladybug.core.logger import pipeline_logger
from scripts.local_file_manager import LocalFileManager


class StorageManager:
    def __init__(self, project, version, local_storage=None, s3_storage=None):
        self.project = project
        self.version = version
        self.local = local_storage
        self.s3 = s3_storage
        self.logger = pipeline_logger 

    # ---------------- FOLDER CREATION ----------------
    def create_folders(self):
        """
        Create project/version folder layout and upload default training configs.

        - Creates the local folder structure if local storage is enabled.
        - Creates S3 folder structure and uploads default training config if S3 storage is enabled.
        - Logs each step of folder creation.
        """
        self.logger.info(f"[{PipelineStep.STORAGE}] Creating project/version folders...")

        if self.local:
            self.local.create_version_folders(model_name=self.project, version=self.version)
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] folders created.")

        if self.s3:
            self.s3.create_version_folders(model_name=self.project, version=self.version)
            self.s3.upload_default_training_config(model_name=self.project, version=self.version)
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] folders created and default training config uploaded.")


    # ---------------- HELPERS ----------------
    def _build(self, stage, filename):
        return {"project": self.project, "version": self.version, "stage": stage, "filename": filename}

    
    # ---------------- SAVE METHODS ----------------
    def save_df(self, df, stage, filename):
        """
        Save a DataFrame to local and/or S3 storage.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            stage (str): Stage/folder to save to (e.g., 'raw_data', 'processed_data').
            filename (str): Name of the file (e.g., 'raw_data.csv').

        Returns:
            bool: True if save attempted (success/failure should be tracked via logs).
        """
        params = self._build(stage, filename)
        self.logger.info(f"[{PipelineStep.STORAGE}] Saving DataFrame '{filename}' to stage='{stage}' ({len(df)} rows)")

        if self.local:
            self.local.upload_df(df=df, **params)
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] DataFrame '{filename}' saved.")

        if self.s3:
            self.s3.upload_df(df=df, **params)
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] DataFrame '{filename}' saved.")

        return True


    def save_json(self, data, stage, filename):
        params = self._build(stage, filename)
        self.logger.debug(f"Saving JSON to stage='{stage}', filename='{filename}'")

        if self.local:
            self.local.upload_json(data=data, **params)
            self.logger.debug("Saved JSON to local storage.")

        if self.s3:
            self.s3.upload_json(data=data, **params)
            self.logger.debug("Saved JSON to S3 storage.")


    def save_jsonl(self, df, stage, filename):
        params = self._build(stage, filename)
        self.logger.debug(f"Saving JSONL to stage='{stage}', filename='{filename}'")

        if self.local:
            self.local.upload_jsonl(df=df, **params)
            self.logger.debug("Saved JSONL to local storage.")

        if self.s3:
            self.s3.upload_jsonl(df=df, **params)
            self.logger.debug("Saved JSONL to S3 storage.")


    def save_yaml(self, data, stage, filename):
        params = self._build(stage, filename)
        self.logger.debug(f"Saving YAML to stage='{stage}', filename='{filename}'")

        if self.local:
            self.local.upload_yaml(data=data, **params)
            self.logger.debug("Saved YAML to local storage.")

        if self.s3:
            self.s3.upload_yaml(data=data, **params)
            self.logger.debug("Saved YAML to S3 storage.")

    

    # ---------------- LOAD METHODS ----------------
    
    def read_default_yaml(self):
        
        default_yaml = LocalFileManager().read_default_yaml()
        self.logger.info(
            f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] "
            f"Default yaml loaded"
        )
        return default_yaml


    def load_df(self, stage: str, filename: str):
        """
        Load a DataFrame from available storage backends.

        Priority:
            1. Local storage (if enabled and file exists)
            2. S3 storage (if enabled)

        Raises:
            FileNotFoundError: If the file does not exist in any backend.
        """
        params = self._build(stage, filename)

        self.logger.info(
            f"[{PipelineStep.STORAGE}] Loading DataFrame "
            f"(stage='{stage}', file='{filename}')"
        )

        if self.local and self.local.exists(**params):
            df = self.local.get_df(**params)
            self.logger.info(
                f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] "
                f"DataFrame loaded ({len(df)} rows)"
            )
            return df

        if self.s3:
            df = self.s3.get_df(**params)
            self.logger.info(
                f"[{PipelineStep.STORAGE}] [{StorageType.S3}] "
                f"DataFrame loaded ({len(df)} rows)"
            )
            return df

        self.logger.error(
            f"[{PipelineStep.STORAGE}] File not found "
            f"(stage='{stage}', file='{filename}')"
        )
        raise FileNotFoundError(
            f"File '{filename}' not found in any storage backend"
        )
