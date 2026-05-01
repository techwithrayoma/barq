from pathlib import Path
import pandas as pd
import yaml
import logging
import json
import os

class LocalFileManager:
    def __init__(self, base_path: str = "/app/ladybug/assets", logger=None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("LocalFileManager")

    # ==========================================================
    # 📁 CREATE VERSION FOLDERS (FIXED + CLEAN)
    # ==========================================================
    def create_version_folders(self, model_name: str, version: str):
        version_path = self.base_path / model_name / version

        folders = [
            "raw_data",
            "processed_data",
            "training",
            "training/training_data",
            "training/training_configs",
            "mlflow",
            "mlflow/artifacts",
            "model",
            "logs",
            "benchmark"
        ]

        for folder in folders:
            folder_path = version_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)

            # marker file so folder is never empty in git/S3 systems
            keep_file = folder_path / ".keep"
            keep_file.touch(exist_ok=True)

            self.logger.info(f"[LOCAL] Created folder: {folder_path}")

    # ==========================================================
    # 📊 DATAFRAME
    # ==========================================================
    def upload_df(self, df, project, version, stage, filename):
        path = self.base_path / project / version / stage / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        if not filename.endswith(".csv"):
            filename += ".csv"
            path = path.with_suffix(".csv")

        df.to_csv(path, index=False)
        self.logger.info(f"[LOCAL] saved df → {path}")
        return str(path)

    def get_df(self, project, version, stage, filename):
        path = self.base_path / project / version / stage / filename
        return pd.read_csv(path)

    def exists(self, project, version, stage, filename):
        path = self.base_path / project / version / stage / filename
        return path.exists()

    # ==========================================================
    # 🧾 JSON
    # ==========================================================
    def upload_json(self, data, project, version, stage, filename):
        path = self.base_path / project / version / stage / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"[LOCAL] saved json → {path}")

    # ==========================================================
    # 🧠 YAML
    # ==========================================================
    def upload_yaml(self, data, project, version, stage, filename):
        path = self.base_path / project / version / stage / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        self.logger.info(f"[LOCAL] saved yaml → {path}")

    # ==========================================================
    # 📥 DEFAULT YAML
    # ==========================================================
    def read_default_yaml(self, filename: str = "default_training.yaml"):
        path = self.base_path / "training_defaults" / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing YAML: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
        

    def folder_exists(self, project, version, stage) -> bool:
        path = os.path.join(self.base_path, project, version, stage)
        return os.path.exists(path) and any(os.scandir(path))
    

    def upload_jsonl(self, df, project, version, stage, filename):
        path = self.base_path / project / version / stage / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        if not filename.endswith(".jsonl"):
            filename += ".jsonl"
            path = path.with_suffix(".jsonl")

        # write JSONL format (1 JSON object per line)
        df.to_json(path, orient="records", lines=True, force_ascii=False)

        self.logger.info(f"[LOCAL] saved jsonl → {path}")
        return str(path)
    

    def get_yaml(self, project, version, stage, filename):
        path = self.base_path / project / version / stage / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing YAML file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)