import logging
import tempfile
import os

from ladybug.enum.pipeline_step import PipelineStep, StorageType
from ladybug.scripts.local_file_manager import LocalFileManager
from ladybug.store.cloud.cloud_provider_factory import CloudProviderFactory
from ladybug.store.llm.llm_provider_factory import LLMProviderFactory
from ladybug.core.load_yaml import load_training_config
from ladybug.scripts.storage_manager import StorageManager
from ladybug.core.logger import pipeline_logger

from .components.data_ingestion import IngestData
from .components.data_clean import DataClean
from .components.data_labeling import DataLabeling
from .components.data_transformation import DataTransformation
from .components.model_training import ModelTraining


class TrainPipeline:
    """
    Orchestrates the machine learning training pipeline.

    Responsibilities:
        - Initialize LLM provider
        - Setup local and/or S3 storage
        - Provide access to pipeline steps:
          ingestion → cleaning → labeling → transformation → training assets → GPU training
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

        # ---------- Logger ----------
        self.logger = pipeline_logger
        self.logger.addFilter(logging.Filter())
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

        # ---------- LLM Provider ----------
        llm_factory = LLMProviderFactory(config)
        self.llm_provider = llm_factory.create(config.LLM_PROVIDER)
        self.llm_provider.set_generation_model(config.OPENAI_MODEL_ID)

        # ---------- Storage Clients ----------
        self.local_storage = LocalFileManager() if store_local else None
        self.s3_storage    = None
        if store_s3:
            factory        = CloudProviderFactory(config)
            self.s3_storage = factory.create_storage("s3")

        if self.local_storage:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.LOCAL}] Local storage enabled")
        if self.s3_storage:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] S3 storage enabled")

        # ---------- Centralized Storage Layer ----------
        self.storage = StorageManager(
            project=self.model_name,
            version=self.version,
            local_storage=self.local_storage,
            s3_storage=self.s3_storage,
        )
        self.logger.info(f"[{PipelineStep.STORAGE}] Storage manager initialized")
        self.logger.info(f"[{PipelineStep.PIPELINE}] TrainPipeline ready")

    # ------------------------------------------------------------------ #
    #  STEP 0 – FOLDERS                                                    #
    # ------------------------------------------------------------------ #
    def create_folders(self):
        self.logger.info(f"[{PipelineStep.STORAGE}] Creating folders...")
        self.storage.create_folders()
        self.logger.info(f"[{PipelineStep.STORAGE}] Folders created.")

    # ------------------------------------------------------------------ #
    #  STEP 1 – DATA INGESTION                                             #
    # ------------------------------------------------------------------ #
    def data_ingestion(self, db):
        self.logger.info(f"[{PipelineStep.INGESTION}] Starting data ingestion...")
        df = IngestData(db=db).fetch_training_data(limit=5)
        self.logger.info(f"[{PipelineStep.INGESTION}] Fetched {len(df)} rows from DB")
        self.storage.save_df(df=df, stage="raw_data", filename="raw_data.csv")
        self.logger.info(f"[{PipelineStep.INGESTION}] {len(df)} rows saved")

    # ------------------------------------------------------------------ #
    #  STEP 2 – DATA CLEAN                                                 #
    # ------------------------------------------------------------------ #
    def data_clean(self):
        self.logger.info(f"[{PipelineStep.CLEAN}] Starting data cleaning...")
        df       = self.storage.load_df(stage="raw_data", filename="raw_data.csv")
        clean_df = DataClean(ingested_data=df).clean_training_data()
        self.storage.save_df(df=clean_df, stage="processed_data", filename="clean_data.csv")
        self.logger.info(f"[{PipelineStep.CLEAN}] Data cleaning complete.")

    # ------------------------------------------------------------------ #
    #  STEP 3 – DATA LABELING                                              #
    # ------------------------------------------------------------------ #
    def data_labeling(self):
        self.logger.info(f"[{PipelineStep.LABELING}] Starting data labeling...")
        df         = self.storage.load_df(stage="processed_data", filename="clean_data.csv")
        labeled_df = DataLabeling(clean_data=df, llm=self.llm_provider).generate_comment_intent()
        self.storage.save_df(df=labeled_df, stage="processed_data", filename="label_data.csv")
        self.logger.info(f"[{PipelineStep.LABELING}] {len(labeled_df)} rows labeled.")

    # ------------------------------------------------------------------ #
    #  STEP 4 – DATA TRANSFORMATION                                        #
    # ------------------------------------------------------------------ #
    def data_transformation(self):
        self.logger.info(f"[{PipelineStep.TRANSFORMATION}] Starting data transformation...")
        df             = self.storage.load_df(stage="processed_data", filename="label_data.csv")
        transformed_df = DataTransformation(labeled_data=df).prepare_llm_finetuning_data()
        self.storage.save_df(df=transformed_df, stage="processed_data", filename="data_transformation.csv")
        self.logger.info(f"[{PipelineStep.TRANSFORMATION}] {len(transformed_df)} rows transformed.")

    # ------------------------------------------------------------------ #
    #  STEP 5 – PREPARE TRAINING ASSETS                                    #
    # ------------------------------------------------------------------ #
    def prepare_training_assets(self):
        df     = self.storage.load_df(stage="processed_data", filename="data_transformation.csv")
        config = load_training_config()

        trainer                    = ModelTraining(df=df, config=config, project=self.model_name, version=self.version)
        train_df, val_df, test_df  = trainer.split_data()

        # ---- JSONL files ----
        self.logger.info(f"[{PipelineStep.TRAINING}] Saving JSONL training files...")
        self.storage.save_jsonl(df=train_df, stage="training/training_data", filename="train.jsonl")
        self.storage.save_jsonl(df=val_df,   stage="training/training_data", filename="val.jsonl")
        self.storage.save_jsonl(df=test_df,  stage="training/training_data", filename="test.jsonl")
        self.logger.info(f"[{PipelineStep.TRAINING}] JSONL files saved.")

        # ---- dataset_info.json ----
        dataset_info = trainer.register_dataset_for_llmfactory()
        self.storage.save_json(data=dataset_info, stage="training/training_data", filename="dataset_info.json")
        self.logger.info(f"[{PipelineStep.TRAINING}] dataset_info.json saved.")

        # ---- final_config.yaml ----
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

    # ------------------------------------------------------------------ #
    #  STEP 6 – RUN MODEL TRAINING  (GPU worker / RunPod)                 #
    # ------------------------------------------------------------------ #
    def run_model_training(self):
        """
        Pull training assets from S3, run LLaMA-Factory fine-tuning,
        then upload the trained adapter back to S3.

        Expected S3 layout (written by prepare_training_assets):
            {project}/{version}/training/training_data/train.jsonl
            {project}/{version}/training/training_data/val.jsonl
            {project}/{version}/training/training_data/dataset_info.json
            {project}/{version}/training/training_configs/final_config.yaml

        Output uploaded to S3:
            {project}/{version}/training/model_output/  (full adapter folder)
        """
        self.logger.info(f"[{PipelineStep.TRAINING}] ===== GPU TRAINING START =====")

        # ----------------------------------------------------------------
        # 1. Load config + build trainer (no DataFrame needed here)
        # ----------------------------------------------------------------
        config  = load_training_config()
        trainer = ModelTraining(
            df=None,          # not needed for training step
            config=config,
            project=self.model_name,
            version=self.version,
        )

        # ----------------------------------------------------------------
        # 2. Download training assets from S3 to a known RunPod path
        #    LLaMA-Factory reads files from disk, so we stage them locally.
        # ----------------------------------------------------------------
        workspace = f"/workspace/{self.model_name}/{self.version}"
        data_dir   = f"{workspace}/data"
        config_dir = f"{workspace}/config"
        output_dir = f"{workspace}/output"

        os.makedirs(data_dir,   exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"[{PipelineStep.TRAINING}] Downloading training assets from S3 → {workspace}")

        # Download JSONL data files
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

        # Download YAML config and patch paths to point at local workspace
        yaml_config = self.storage.load_yaml(
            stage="training/training_configs",
            filename="final_config.yaml",
        )
        yaml_config["dataset_dir"] = data_dir
        yaml_config["output_dir"]  = output_dir

        config_path = f"{config_dir}/final_config.yaml"
        self.storage.write_yaml_to_disk(data=yaml_config, local_path=config_path)

        self.logger.info(f"[{PipelineStep.TRAINING}] Assets staged. Launching LLaMA-Factory...")

        # ----------------------------------------------------------------
        # 3. Run training — blocks until done
        # ----------------------------------------------------------------
        trainer.run_llamafactory_training(config_path=config_path)

        # ----------------------------------------------------------------
        # 4. Upload trained adapter back to S3
        # ----------------------------------------------------------------
        self.logger.info(f"[{PipelineStep.TRAINING}] Uploading model adapter to S3...")
        self.storage.upload_folder(
            local_dir=output_dir,
            stage="training/model_output",
        )
        self.logger.info(f"[{PipelineStep.TRAINING}] Model adapter uploaded to S3.")
        self.logger.info(f"[{PipelineStep.TRAINING}] ===== GPU TRAINING COMPLETE =====")