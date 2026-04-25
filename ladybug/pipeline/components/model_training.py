import os
import json
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from ladybug.core.logger import pipeline_logger

logger = pipeline_logger


class ModelTraining:
    """
    Handles all model training preparation and execution.

    Responsibilities:
        - Split data into train/val/test sets
        - Register dataset for LLaMA-Factory
        - Build training config YAML
        - Run LLaMA-Factory fine-tuning via CLI
    """

    LABEL_MAP = {
        "complaint":   "Complaint",
        "question":    "Question",
        "suggestion":  "Suggestion",
        "statement":   "Statement",
        "praise":      "Praise",
    }

    def __init__(self, df: pd.DataFrame, config: dict, project: str, version: str):
        """
        Args:
            df      : Transformed DataFrame with columns [instruction, input, output]
            config  : Training config dict loaded from YAML
            project : Model/project name  (e.g. "ladybug")
            version : Pipeline version    (e.g. "v1")
        """
        self.df      = df
        self.config  = config
        self.project = project
        self.version = version

    # ------------------------------------------------------------------ #
    #  DATA SPLIT                                                          #
    # ------------------------------------------------------------------ #
    def split_data(
        self,
        test_size:  float = 0.10,
        val_size:   float = 0.10,
        random_state: int = 42,
    ):
        """
        Split DataFrame into train / val / test.

        Returns:
            train_df, val_df, test_df  (pandas DataFrames)
        """
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df["output"] if "output" in self.df.columns else None,
        )

        relative_val = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            random_state=random_state,
            stratify=train_val_df["output"] if "output" in train_val_df.columns else None,
        )

        logger.info(
            f"[ModelTraining] Split — train={len(train_df)} | "
            f"val={len(val_df)} | test={len(test_df)}"
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------ #
    #  DATASET REGISTRATION (LLaMA-Factory dataset_info.json format)      #
    # ------------------------------------------------------------------ #
    def register_dataset_for_llmfactory(self) -> dict:
        """
        Build the dataset_info.json entry required by LLaMA-Factory.

        LLaMA-Factory needs to know column names so it can read the JSONL files.

        Returns:
            dict – ready to be serialised as dataset_info.json
        """
        dataset_info = {
            f"{self.project}_{self.version}_train": {
                "file_name": "train.jsonl",
                "columns": {
                    "prompt":   "instruction",
                    "query":    "input",
                    "response": "output",
                },
            },
            f"{self.project}_{self.version}_val": {
                "file_name": "val.jsonl",
                "columns": {
                    "prompt":   "instruction",
                    "query":    "input",
                    "response": "output",
                },
            },
        }
        logger.info("[ModelTraining] dataset_info.json built")
        return dataset_info

    # ------------------------------------------------------------------ #
    #  YAML CONFIG BUILDER                                                 #
    # ------------------------------------------------------------------ #
    def build_llm_training_config(
        self,
        default_config: dict,
        user_overrides: dict | None = None,
    ) -> dict:
        """
        Merge default YAML config with user overrides and inject
        dataset / output paths derived from project + version.

        Args:
            default_config : Base config loaded from storage
            user_overrides : Key-value pairs that overwrite defaults

        Returns:
            Final merged config dict
        """
        config = dict(default_config)

        # ---- inject dataset name so LLaMA-Factory finds the right entry ----
        config["dataset"] = f"{self.project}_{self.version}_train"
        config["val_size"] = 0.1          # LLaMA-Factory reads this too

        # ---- output dir: RunPod writes here, we upload to S3 afterwards ----
        config["output_dir"] = f"/workspace/outputs/{self.project}/{self.version}"

        # ---- apply caller overrides last so they win ----
        if user_overrides:
            config.update(user_overrides)

        logger.info(f"[ModelTraining] Final YAML config built: {list(config.keys())}")
        return config

    # ------------------------------------------------------------------ #
    #  RUN LLAMA-FACTORY TRAINING                                          #
    # ------------------------------------------------------------------ #
    def run_llamafactory_training(self, config_path: str) -> None:
        """
        Execute LLaMA-Factory fine-tuning via CLI.

        This is called by TrainPipeline.run_model_training() on the RunPod
        GPU worker.  It blocks until training completes (or raises on failure).

        Args:
            config_path : Absolute path to the final_config.yaml on disk
                          (already downloaded from S3 by StorageManager)

        Raises:
            RuntimeError : if llamafactory-cli exits with a non-zero code
        """
        cmd = ["llamafactory-cli", "train", config_path]

        logger.info(f"[ModelTraining] Launching: {' '.join(cmd)}")

        # Stream stdout/stderr live so Celery logs show training progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            logger.info(f"[LLaMA-Factory] {line.rstrip()}")

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f"llamafactory-cli exited with code {process.returncode}"
            )

        logger.info("[ModelTraining] LLaMA-Factory training finished successfully")