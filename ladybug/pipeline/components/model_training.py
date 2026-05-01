import os
import json
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from ladybug.core.logger import pipeline_logger
import mlflow
import time


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
            val_size: float = 0.1,
            random_state: int = 42,
            ensure_all_classes_in_val: bool = True,
        ):
            """
            Split DataFrame into train / validation.

            Features:
            - Stratified split (if possible)
            - Handles small datasets
            - Verifies class distribution
            - Optionally guarantees all classes in validation
            """

            # -----------------------------
            # Basic checks
            # -----------------------------
            if "output" not in self.df.columns:
                raise ValueError("DataFrame must contain 'output' column")

            df = self.df.copy()

            # -----------------------------
            # Class distribution BEFORE
            # -----------------------------
            logger.info("📊 Full dataset distribution:")
            logger.info(f"\n{df['output'].value_counts()}")

            # -----------------------------
            # Decide stratification
            # -----------------------------
            stratify_col = df["output"]

            if len(df) < 50 or df["output"].value_counts().min() < 2:
                stratify_col = None
                logger.warning("⚠️ Dataset too small or rare classes → disabling stratification")

            # -----------------------------
            # Split
            # -----------------------------
            train_df, val_df = train_test_split(
                df,
                test_size=val_size,
                random_state=random_state,
                stratify=stratify_col,
            )

            # -----------------------------
            # Ensure ALL classes in val
            # -----------------------------
            if ensure_all_classes_in_val:
                missing_classes = set(df["output"]) - set(val_df["output"])

                if missing_classes:
                    logger.warning(f"⚠️ Missing classes in val: {missing_classes}")
                    logger.warning("🔁 Fixing validation set to include all classes...")

                    # Move one sample per missing class from train → val
                    for cls in missing_classes:
                        sample = train_df[train_df["output"] == cls].sample(
                            n=1, random_state=random_state
                        )
                        val_df = val_df.append(sample)
                        train_df = train_df.drop(sample.index)

            # -----------------------------
            # Final distribution check
            # -----------------------------
            logger.info("\n📊 Train distribution:")
            logger.info(f"\n{train_df['output'].value_counts()}")

            logger.info("\n📊 Validation distribution:")
            logger.info(f"\n{val_df['output'].value_counts()}")

            logger.info(
                f"\n✅ Final Split — train={len(train_df)} | val={len(val_df)}"
            )

            return train_df.reset_index(drop=True), val_df.reset_index(drop=True)



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


    def run_llamafactory_training(self, config_path: str):
        
        mlflow.set_tracking_uri("http://0.0.0.0:57321")

        mlflow.set_experiment(f"{self.project}-{self.version}")

        start_time = time.time()

        with mlflow.start_run() as run:

            run_id = run.info.run_id

            # ----------------------------
            # BASIC PARAMS
            # ----------------------------
            mlflow.log_param("project", self.project)
            mlflow.log_param("version", self.version)


            # ----------------------------
            # 🔥 CONFIG LOGGING
            # ----------------------------
            mlflow.log_dict(self.config, "training_config.json")

            # ----------------------------
            # RUN TRAINING
            # ----------------------------
            cmd = ["llamafactory-cli", "train", config_path]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in process.stdout:
                logger.info(line)

            process.wait()

            if process.returncode != 0:
                raise RuntimeError("Training failed")

            # ----------------------------
            # METRICS
            # ----------------------------
            duration_sec = time.time() - start_time
            gpu_hours = duration_sec / 3600
            cost = gpu_hours * 1.5  # adjust later

            mlflow.log_metric("training_time_sec", duration_sec)
            mlflow.log_metric("gpu_hours", gpu_hours)
            mlflow.log_metric("cost_usd", cost)

            logger.info(f"[MLFLOW] Run ID: {run_id}")

            return run_id

        






### ignore for now 
# def _save_model_metadata(self, db, training_cutoff, total_cost):
#     from models import Model

#     model = Model(
#         model_name=self.model_name,
#         model_version=self.version,
#         training_run_type="runpod",
#         dataset_dvc_hash=None,  # or later
#         total_training_cost_usd=total_cost,
#         gpu_type="A100",  # example
#         gpu_hours=1.5,    # example
#         created_at=datetime.now(timezone.utc)
#     )

#     db.add(model)
#     db.commit()


# a snapshot of EXACT training dataset used for a model run
# def create_dvc_snapshot(self, train_df, val_df, storage_path: str):
#     """
#     Create a reproducible dataset fingerprint (DVC-style hash).
#     """

#     def hash_df(df):
#         # stable hash of dataset content
#         return hashlib.sha256(
#             pd.util.hash_pandas_object(df, index=True).values
#         ).hexdigest()

#     train_hash = hash_df(train_df)
#     val_hash   = hash_df(val_df)

#     dvc_snapshot = {
#         "project": self.project,
#         "version": self.version,
#         "train_hash": train_hash,
#         "val_hash": val_hash,
#         "rows_train": len(train_df),
#         "rows_val": len(val_df),
#     }

#     # save locally (or via storage manager)
#     path = os.path.join(storage_path, "dvc_snapshot.json")

#     with open(path, "w") as f:
#         json.dump(dvc_snapshot, f, indent=2)

#     logger.info(f"[DVC] Snapshot created: {dvc_snapshot}")

#     return dvc_snapshot