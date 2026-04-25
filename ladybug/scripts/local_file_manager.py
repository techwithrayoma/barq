import os
from pathlib import Path
import yaml
import logging

class LocalFileManager:
    def __init__(self, base_path: str = "/app/ladybug/assets", logger: logging.Logger = None):
        """
        base_path: the root folder where all model/version folders will be created.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("LocalFileManager")
        self.logger.info(f"LocalFileManager initialized at base path: {self.base_path}")

    def create_version_folders(self, model_name: str, version: str):
        version_path = self.base_path / model_name / version

        folders = [
            "raw_data",
            "processed_data",
            "training/",
            "mlflow",
            "mlflow/artifacts",
            "model",
            "logs"
        ]

        for folder in folders:
            folder_path = version_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            keep_file = folder_path / ".keep"
            keep_file.touch(exist_ok=True)
            self.logger.info(f"Created folder: {folder_path} with .keep file")

    def upload_df(self, df, project: str, version: str, stage: str, filename: str):
        """
        Saves dataframe locally following structure:
        {base_path}/{project}/{version}/{stage}/{filename}
        """
        stage_path = self.base_path / project / version / stage
        stage_path.mkdir(parents=True, exist_ok=True)

        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"

        file_path = stage_path / filename
        df.to_csv(file_path, index=False)
        self.logger.info(f"Saved dataframe locally at: {file_path}")
        return str(file_path)

    def read_default_yaml(self, filename: str = "default_training.yaml"):
        """
        Reads the default training YAML from the assets folder.
        Returns a Python dict.
        """
        yaml_path = self.base_path / "training_defaults" / filename
        if not yaml_path.exists():
            self.logger.error(f"Training template not found: {yaml_path}")
            raise FileNotFoundError(f"Training template not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded default YAML from: {yaml_path}")
        return config
