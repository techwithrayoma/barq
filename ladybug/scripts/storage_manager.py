import os
import yaml
from pathlib import Path
import pandas as pd

from ladybug.enum.pipeline_step import PipelineStep, StorageType
from ladybug.core.logger import pipeline_logger
from ladybug.scripts.local_file_manager import LocalFileManager


class StorageManager:
    """
    Single interface for all pipeline storage operations.

    Priority:   LOCAL cache first  →  S3 fallback
    Protection: Never overwrites local cache (saves cost + enables resumption)
    """

    def __init__(self, project, version, local_storage=None, s3_storage=None):
        self.project = project
        self.version = version
        self.local   = local_storage
        self.s3      = s3_storage
        self.logger  = pipeline_logger

    # ────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ────────────────────────────────────────────────────────────────────────

    def _build(self, stage, filename):
        """Build the standard kwargs dict used by every storage client."""
        return {
            "project":  self.project,
            "version":  self.version,
            "stage":    stage,
            "filename": filename,
        }

    # ────────────────────────────────────────────────────────────────────────
    # EXISTS CHECK  ← fixes the missing _exists bug in TrainPipeline
    # ────────────────────────────────────────────────────────────────────────

    def exists(self, stage: str, filename: str) -> bool:
        """
        Return True if the artifact already exists anywhere in storage.
        LOCAL is checked first (cheaper + faster).
        """
        params = self._build(stage, filename)

        if self.local and self.local.exists(**params):
            return True

        if self.s3 and self.s3.exists(**params):
            return True

        return False


    def folder_exists(self, stage: str) -> bool:
        """
        Check if any file exists inside a stage folder.
        Works for both local + S3.
        """

        # LOCAL
        if self.local:
            if self.local.folder_exists(
                project=self.project,
                version=self.version,
                stage=stage,
            ):
                return True

        # S3
        if self.s3:
            prefix = f"{self.project}/{self.version}/{stage}/"

            response = self.s3.s3_client.list_objects_v2(
                Bucket=self.s3.S3_BUCKET_NAME,
                Prefix=prefix,
                MaxKeys=1,  # ⚡ FAST check
            )

            if "Contents" in response:
                return True

        return False


    # ────────────────────────────────────────────────────────────────────────
    # FOLDER CREATION
    # ────────────────────────────────────────────────────────────────────────

    def create_folders(self):
        self.logger.info(f"[{PipelineStep.STORAGE}] Creating project/version folders...")

        if self.local:
            self.local.create_version_folders(model_name=self.project, version=self.version)
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] folders created.")

        if self.s3:
            self.s3.create_version_folders(model_name=self.project, version=self.version)

            if hasattr(self.s3, "copy_benchmark_to_version"):
                self.s3.copy_benchmark_to_version(
                    model_name=self.project,
                    version=self.version,
                )

            self.s3.upload_default_training_config(
                model_name=self.project,
                version=self.version,
            )
            self.logger.info(
                f"[{PipelineStep.STORAGE}] [{StorageType.S3}] folders created and config uploaded."
            )

    # ────────────────────────────────────────────────────────────────────────
    # SAVE: DataFrame
    # ────────────────────────────────────────────────────────────────────────

    def save_df(self, df, stage, filename):
        params = self._build(stage, filename)

        self.logger.info(
            f"[{PipelineStep.STORAGE}] Saving DataFrame '{filename}' "
            f"stage='{stage}' ({len(df)} rows)"
        )

        if self.local:
            # Cache protection: never overwrite if already saved
            if not self.local.exists(**params):
                self.local.upload_df(df=df, **params)
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] saved.")

        if self.s3:
            self.s3.upload_df(df=df, **params)
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] saved.")

    # ────────────────────────────────────────────────────────────────────────
    # SAVE: JSON
    # ────────────────────────────────────────────────────────────────────────

    def save_json(self, data, stage, filename):
        params = self._build(stage, filename)

        if self.local and not self.local.exists(**params):
            self.local.upload_json(data=data, **params)

        if self.s3:
            self.s3.upload_json(data=data, **params)

    # ────────────────────────────────────────────────────────────────────────
    # SAVE: JSONL
    # ────────────────────────────────────────────────────────────────────────

    def save_jsonl(self, df, stage, filename):
        params = self._build(stage, filename)

        if self.local and not self.local.exists(**params):
            self.local.upload_jsonl(df=df, **params)

        if self.s3:
            self.s3.upload_jsonl(df=df, **params)

    # ────────────────────────────────────────────────────────────────────────
    # SAVE: YAML
    # ────────────────────────────────────────────────────────────────────────

    def save_yaml(self, data, stage, filename):
        params = self._build(stage, filename)

        if self.local and not self.local.exists(**params):
            self.local.upload_yaml(data=data, **params)

        if self.s3:
            self.s3.upload_yaml(data=data, **params)

    # ────────────────────────────────────────────────────────────────────────
    # LOAD: DataFrame  (local first → S3 fallback)
    # ────────────────────────────────────────────────────────────────────────

    def load_df(self, stage: str, filename: str) -> pd.DataFrame:
        params = self._build(stage, filename)

        self.logger.info(f"[{PipelineStep.STORAGE}] Loading DataFrame {stage}/{filename}")

        # 1. Local cache
        if self.local and self.local.exists(**params):
            df = self.local.get_df(**params)
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] loaded ({len(df)} rows)")
            return df

        # 2. S3 fallback
        if self.s3:
            df = self.s3.get_df(**params)

            # Cache locally so the next load is instant
            if self.local:
                self.local.upload_df(df=df, **params)

            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] loaded ({len(df)} rows)")
            return df

        raise FileNotFoundError(f"DataFrame not found: {stage}/{filename}")

    # ────────────────────────────────────────────────────────────────────────
    # LOAD: Default YAML template
    # ────────────────────────────────────────────────────────────────────────

    def read_default_yaml(self):
        # Use self.local if available; fall back to a plain LocalFileManager
        manager = self.local if self.local else LocalFileManager()
        data    = manager.read_default_yaml()
        self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] default yaml loaded")
        return data

    # ────────────────────────────────────────────────────────────────────────
    # LOAD: YAML from storage
    # ────────────────────────────────────────────────────────────────────────

    def load_yaml(self, stage: str, filename: str) -> dict:
        """
        Load a YAML file.  Local first → S3 fallback.
        Returns a plain Python dict.
        """
        params = self._build(stage, filename)

        # Local
        if self.local and self.local.exists(**params):
            return self.local.get_yaml(**params)

        # S3 fallback
        if self.s3:
            return self.s3.get_yaml(**params)

        raise FileNotFoundError(f"YAML not found: {stage}/{filename}")

    # ────────────────────────────────────────────────────────────────────────
    # WRITE: YAML directly to disk (used during GPU training on RunPod)
    # ────────────────────────────────────────────────────────────────────────

    def write_yaml_to_disk(self, data: dict, path: str):
        """
        Write a YAML dict to an absolute local path.
        Used in run_model_training() to write the final config for LLaMA-Factory.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        self.logger.info(f"[{PipelineStep.STORAGE}] YAML written to disk: {path}")

    # ────────────────────────────────────────────────────────────────────────
    # DOWNLOAD: single file from S3 to local path
    # ────────────────────────────────────────────────────────────────────────

    def download_file(self, stage: str, filename: str, local_path: str):
        """
        Download a single file from S3 to an absolute local path.
        Used in run_model_training() to pull data/config to RunPod workspace.
        """
        if not self.s3:
            raise RuntimeError("download_file requires S3 storage to be enabled.")

        key = f"{self.project}/{self.version}/{stage}/{filename}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        self.s3.s3_client.download_file(
            Bucket=self.s3.S3_BUCKET_NAME,
            Key=key,
            Filename=local_path,
        )
        self.logger.info(
            f"[{PipelineStep.STORAGE}] [{StorageType.S3}] "
            f"Downloaded {stage}/{filename} → {local_path}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # UPLOAD: entire local folder to S3  (model adapter upload)
    # ────────────────────────────────────────────────────────────────────────

    def upload_folder(self, local_dir: str, stage: str):
        """
        Recursively upload every file in local_dir to S3 under stage/.
        Used after training to persist the LoRA adapter.
        """
        if not self.s3:
            raise RuntimeError("upload_folder requires S3 storage to be enabled.")

        local_root = Path(local_dir)

        if not local_root.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        uploaded = 0
        for file_path in local_root.rglob("*"):
            if not file_path.is_file():
                continue

            relative = file_path.relative_to(local_root)
            s3_key   = f"{self.project}/{self.version}/{stage}/{relative}"

            self.s3.s3_client.upload_file(
                Filename=str(file_path),
                Bucket=self.s3.S3_BUCKET_NAME,
                Key=s3_key,
            )
            uploaded += 1

        self.logger.info(
            f"[{PipelineStep.STORAGE}] [{StorageType.S3}] "
            f"Uploaded {uploaded} files from {local_dir} → s3://{stage}/"
        )

    # ────────────────────────────────────────────────────────────────────────
    # LOAD: Benchmark DataFrame
    # ────────────────────────────────────────────────────────────────────────

    def load_benchmark(self) -> pd.DataFrame:
        """
        Load benchmark CSV.  Local first → S3 fallback.
        """
        # Local cache
        if self.local:
            try:
                return self.local.get_df(
                    project=self.project,
                    version=self.version,
                    stage="benchmark",
                    filename="benchmark.csv",
                )
            except Exception:
                pass

        # S3 fallback — uses get_df (consistent with the rest of the codebase)
        if self.s3:
            return self.s3.get_df(
                project=self.project,
                version=self.version,
                stage="benchmark",
                filename="benchmark.csv",
            )

        raise FileNotFoundError("No benchmark data available in local or S3 storage.")