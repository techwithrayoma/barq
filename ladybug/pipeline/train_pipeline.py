from datetime import datetime
import logging
import os

from ladybug.enum.pipeline_step import PipelineStep, StorageType
from ladybug.scripts.local_file_manager import LocalFileManager
from ladybug.store.cloud.cloud_provider_factory import CloudProviderFactory
from ladybug.store.llm.llm_provider_factory import LLMProviderFactory
from ladybug.core.load_yaml import load_training_config
from ladybug.scripts.storage_manager import StorageManager
from ladybug.core.logger import pipeline_logger
from ladybug.repository.model_prediction_repository import ModelPredictionRepository

from .components.data_ingestion import IngestData
from .components.data_clean import DataClean
from .components.data_labeling import DataLabeling
from .components.data_transformation import DataTransformation
from .components.model_training import ModelTraining
from .components.model_evaluation import ModelEvaluation


class TrainPipeline:
    """
    Orchestrates the machine learning training pipeline.

    Steps:
        0. create_folders
        1. data_ingestion        → raw_data/raw_data.csv
        2. data_clean            → processed_data/clean_data.csv
        3. data_labeling         → processed_data/label_data.csv       [cached]
        4. data_transformation   → processed_data/data_transformation.csv [cached]
        5. prepare_training_assets → training/ JSONL + dataset_info + config [cached]
        6. run_model_training    → GPU training + evaluation + upload
    """

    def __init__(
        self,
        config,
        version: str,
        store_s3: bool = True,
        store_local: bool = False,
        model_name: str = "ladybug",
        task_id: str = "no-task",
    ):
        self.version    = version
        self.model_name = model_name

        # ── Logger ──────────────────────────────────────────────────────────
        self.logger = pipeline_logger
        self.logger.addFilter(
            type(
                "TaskFilter",
                (),
                {"filter": lambda self, record: setattr(record, "task_id", task_id) or True},
            )()
        )
        self.logger.info(
            f"[{PipelineStep.PIPELINE}] Initializing pipeline "
            f"model='{self.model_name}' version='{self.version}'"
        )

        # ── LLM Provider ────────────────────────────────────────────────────
        llm_factory = LLMProviderFactory(config)
        self.llm_provider = llm_factory.create(config.LLM_PROVIDER)
        self.llm_provider.set_generation_model(config.OPENAI_MODEL_ID)

        # ── Storage Clients ─────────────────────────────────────────────────
        self.local_storage = LocalFileManager() if store_local else None
        self.s3_storage    = None
        if store_s3:
            factory         = CloudProviderFactory(config)
            self.s3_storage = factory.create_storage("s3")

        if self.local_storage:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] Local storage enabled")
        if self.s3_storage:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] S3 storage enabled")


        # ── Centralised Storage Layer ────────────────────────────────────────
        self.storage = StorageManager(
            project=self.model_name,
            version=self.version,
            local_storage=self.local_storage,
            s3_storage=self.s3_storage,
        )
        self.logger.info(f"[{PipelineStep.STORAGE}] StorageManager ready")
        self.logger.info(f"[{PipelineStep.PIPELINE}] TrainPipeline ready")

    # ────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ────────────────────────────────────────────────────────────────────────

    def _exists(self, stage: str, filename: str) -> bool:
        """Return True if the artifact already exists in local or S3 storage."""
        return self.storage.exists(stage=stage, filename=filename)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 0 – FOLDERS
    # ────────────────────────────────────────────────────────────────────────

    def create_folders(self):
        # CACHE GUARD — skip if version already initialized
        if self.storage.folder_exists("training") or self.storage.folder_exists("raw_data"):
            self.logger.info(
                f"[{PipelineStep.STORAGE}] [CACHE] folders already exist — skipping creation."
            )
            return


        self.logger.info(f"[{PipelineStep.STORAGE}] Creating folders...")
        self.storage.create_folders()
        self.logger.info(f"[{PipelineStep.STORAGE}] Folders ready.")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1 – DATA INGESTION
    # ────────────────────────────────────────────────────────────────────────
    def data_ingestion(self, db):
        # CACHE GUARD — skip if raw data already saved
        if self._exists("raw_data", "raw_data.csv"):
            self.logger.info(f"[{PipelineStep.INGESTION}] [CACHE] raw_data.csv already exists — skipping ingestion.")
            return

        self.logger.info(f"[{PipelineStep.INGESTION}] Starting data ingestion...")

        df = IngestData(db=db).fetch_training_data(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2026, 1, 1)
        )

        self.logger.info(f"[{PipelineStep.INGESTION}] Fetched {len(df)} rows from DB")
        self.storage.save_df(df=df, stage="raw_data", filename="raw_data.csv")
        self.logger.info(f"[{PipelineStep.INGESTION}] {len(df)} rows saved.")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2 – DATA CLEAN
    # ────────────────────────────────────────────────────────────────────────

    def data_clean(self):
        # CACHE GUARD — skip if clean data already saved
        if self._exists("processed_data", "clean_data.csv"):
            self.logger.info(f"[{PipelineStep.CLEAN}] [CACHE] clean_data.csv already exists — skipping cleaning.")
            return

        self.logger.info(f"[{PipelineStep.CLEAN}] Starting data cleaning...")

        df       = self.storage.load_df(stage="raw_data", filename="raw_data.csv")
        clean_df = DataClean(ingested_data=df).clean_training_data()

        self.storage.save_df(df=clean_df, stage="processed_data", filename="clean_data.csv")
        self.logger.info(f"[{PipelineStep.CLEAN}] Data cleaning complete. {len(clean_df)} rows saved.")


    # ────────────────────────────────────────────────────────────────────────
    # STEP 3 – DATA LABELING
    # ────────────────────────────────────────────────────────────────────────
    def data_labeling(self, db):
        # CACHE GUARD
        if self._exists("processed_data", "label_data.csv"):
            self.logger.info(
                f"[{PipelineStep.LABELING}] [CACHE] label_data.csv already exists — skipping labeling."
            )
            return

        self.logger.info(f"[{PipelineStep.LABELING}] Starting data labeling...")

        # LOAD CLEAN DATA
        df = self.storage.load_df(
            stage="processed_data",
            filename="clean_data.csv"
        )

        # RUN LABELING
        labeled_df, db_records, cost_tracker = DataLabeling(
            clean_data=df,
            llm=self.llm_provider,
        ).generate_comment_intent()

        # SAVE DATASET (MAIN ARTIFACT)
        self.storage.save_df(
            df=labeled_df,
            stage="processed_data",
            filename="label_data.csv",
        )

        # SAVE COST + METADATA (IMPORTANT FOR MODEL TRACKING)
        metadata = {
            "cost_tracker": cost_tracker,
            "rows": len(labeled_df),
            "model_name": self.model_name,
            "version": self.version,
            "stage": "labeling",
        }

        self.storage.save_json(
            data=metadata,
            stage="processed_data",
            filename="label_data_metadata.json",
        )

        # Need this to be linked
        # prediction_repo = ModelPredictionRepository(db=db)

        # # SAVE LABELS TO DB (for audit / tracking)
        # prediction_repo.insert_predictions_batch(db_records)


        # LOG SUMMARY
        self.logger.info(
            f"[{PipelineStep.LABELING}] Labeling complete — "
            f"rows={len(labeled_df)}, "
            f"cost=${cost_tracker['total_cost']:.4f}"
        )



    # ────────────────────────────────────────────────────────────────────────
    # STEP 4 – DATA TRANSFORMATION
    # ────────────────────────────────────────────────────────────────────────
    def data_transformation(self):
        # CACHE GUARD
        if self._exists("processed_data", "data_transformation.csv"):
            self.logger.info(f"[{PipelineStep.TRANSFORMATION}] [CACHE] data_transformation.csv already exists — skipping.")
            return

        self.logger.info(f"[{PipelineStep.TRANSFORMATION}] Starting data transformation...")

        df = self.storage.load_df(stage="processed_data", filename="label_data.csv")

        transformed_df = DataTransformation(df).prepare_llm_finetuning_data()

        self.storage.save_df(
            df=transformed_df,
            stage="processed_data",
            filename="data_transformation.csv",
        )
        self.logger.info(f"[{PipelineStep.TRANSFORMATION}] Transformation complete. {len(transformed_df)} rows saved.")

    
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 5 – PREPARE TRAINING ASSETS
    # ────────────────────────────────────────────────────────────────────────
    def prepare_training_assets(self):
        # CACHE GUARD — all three artifacts must exist to skip
        all_cached = (
            self._exists("training/training_data",    "train.jsonl")
            and self._exists("training/training_data",    "val.jsonl")
            and self._exists("training/training_data",    "dataset_info.json")
            and self._exists("training/training_configs", "final_config.yaml")
        )
        if all_cached:
            self.logger.info(f"[{PipelineStep.TRAINING}] [CACHE] Training assets already exist — skipping preparation.")
            return

        self.logger.info(f"[{PipelineStep.TRAINING}] Preparing training assets...")

        df     = self.storage.load_df(stage="processed_data", filename="data_transformation.csv")
        config = load_training_config()

        trainer          = ModelTraining(df=df, config=config, project=self.model_name, version=self.version)
        train_df, val_df = trainer.split_data()

        # ── JSONL files ──────────────────────────────────────────────────────
        self.storage.save_jsonl(df=train_df, stage="training/training_data", filename="train.jsonl")
        self.storage.save_jsonl(df=val_df,   stage="training/training_data", filename="val.jsonl")
        self.logger.info(f"[{PipelineStep.TRAINING}] JSONL files saved.")

        # ── dataset_info.json ────────────────────────────────────────────────
        dataset_info = trainer.register_dataset_for_llmfactory()
        self.storage.save_json(data=dataset_info, stage="training/training_data", filename="dataset_info.json")
        self.logger.info(f"[{PipelineStep.TRAINING}] dataset_info.json saved.")

        # ── final_config.yaml ────────────────────────────────────────────────
        default_yaml = self.storage.read_default_yaml()
        user_overrides = {
            "learning_rate":    5e-5,
            "num_train_epochs": 5.0,
        }
        final_config = trainer.build_llm_training_config(
            default_config=default_yaml,
            user_overrides=user_overrides,
        )
        self.storage.save_yaml(data=final_config, stage="training/training_configs", filename="final_config.yaml")
        self.logger.info(f"[{PipelineStep.TRAINING}] final_config.yaml saved.")
        self.logger.info(f"[{PipelineStep.TRAINING}] All training assets ready.")

    
    
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 6 – RUN GPU TRAINING (runs on RunPod)
    # ────────────────────────────────────────────────────────────────────────
    def run_model_training(self):
        
        self.logger.info(f"[{PipelineStep.TRAINING}] ===== GPU TRAINING START =====")


        workspace  = f"/workspace/{self.model_name}/{self.version}"
        data_dir   = f"{workspace}/data"
        config_dir = f"{workspace}/config"
        output_dir = f"{workspace}/output"

        os.makedirs(data_dir,   exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)


        # ── Download training files from S3 ─────────────────────────────────
        self.logger.info(f"[{PipelineStep.TRAINING}] Downloading training files from S3...")
        
        self.storage.download_file(
            stage="training/training_data",
            filename="train.jsonl",
            local_path=f"{data_dir}/train.jsonl",
        )
        self.storage.download_file(
            stage="training/training_data",
            filename="val.jsonl",
            local_path=f"{data_dir}/val.jsonl",
        )
        self.storage.download_file(
            stage="training/training_data",
            filename="dataset_info.json",
            local_path=f"{data_dir}/dataset_info.json",
        )

        self.logger.info(f"[{PipelineStep.TRAINING}] Training files downloaded.")


        # ── Build final config with RunPod-specific paths ────────────────────
        yaml_config = self.storage.load_yaml(
            stage="training/training_configs",
            filename="final_config.yaml",
        )
        yaml_config["dataset_dir"] = data_dir
        yaml_config["output_dir"]  = output_dir

        config_path = f"{config_dir}/final_config.yaml"
        self.storage.write_yaml_to_disk(yaml_config, config_path)


        trainer = ModelTraining(
            df=None,
            config=yaml_config,
            project=self.model_name,
            version=self.version,
        )

        # ── Run LLaMA-Factory training ───────────────────────────────────────
        run_id = trainer.run_llamafactory_training(config_path=config_path)
        self.logger.info(f"[{PipelineStep.TRAINING}] Training complete. MLflow run_id={run_id}")


        # ── Upload model adapter to S3 ───────────────────────────────────────
        self.logger.info(f"[{PipelineStep.TRAINING}] Uploading model adapter to S3...")
        self.storage.upload_folder(
            local_dir=output_dir,
            stage="training/model_output",
        )


        # # ── Evaluation ──────────────────────────────────────────────────────
        # self.logger.info(f"[{PipelineStep.EVALUATION}] ===== EVALUATION START =====")

        # benchmark_df = self.storage.load_benchmark()

        # evaluator = ModelEvaluation(
        #     model=trainer,
        #     benchmark_df=benchmark_df,   # pass directly — no file needed
        # )

        # metrics = evaluator.evaluate()

        # self.storage.save_json(
        #     data=metrics,
        #     stage="mlflow",
        #     filename="evaluation_metrics.json",
        # )
        # self.logger.info(
        #     f"[{PipelineStep.EVALUATION}] accuracy={metrics['accuracy']:.4f} "
        #     f"f1={metrics['f1_score']:.4f}"
        # )
        # self.logger.info(f"[{PipelineStep.TRAINING}] ===== COMPLETE =====")