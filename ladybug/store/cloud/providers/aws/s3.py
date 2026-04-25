import io
import boto3
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import json
import yaml
import io

from ladybug.enum.pipeline_step import PipelineStep, StorageType
from ladybug.core.logger import pipeline_logger


class S3():
    def __init__(self, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME):
        self.AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID
        self.AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY
        self.AWS_REGION = AWS_REGION
        self.S3_BUCKET_NAME = S3_BUCKET_NAME


        self.logger = pipeline_logger

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_REGION
        )

 
    # ---------------- SAVE METHODS ----------------   
    def upload_jsonl(self, df, project, version, stage, filename):
        """
        Upload a DataFrame as JSONL to S3.
        Each row is a JSON object, lines separated by newline.
        """
        key = f"{project}/{version}/{stage}/{filename}"

        buffer = io.StringIO()
        df.to_json(buffer, orient="records", lines=True, force_ascii=False)
        buffer.seek(0)

        try:
            self.s3_client.put_object(
                Bucket=self.S3_BUCKET_NAME,
                Key=key,
                Body=buffer.getvalue()
            )
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Uploaded {filename} to S3 at {key}")
        except Exception as e:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Failed to upload at {key}: {e}")

    
    def upload_df(self, df, project, version, stage, filename):
        """
        Upload a pandas DataFrame to S3 storage.

        Args:
            df (pd.DataFrame): The DataFrame to upload.
            project (str): The project name to organize storage paths.
            version (str): The version identifier for the project.
            stage (str): The stage or folder under the project/version (e.g., "raw_data").
            filename (str): The target filename to save the DataFrame as (e.g., "raw_data.csv").

        Behavior:
            - Converts the DataFrame to CSV in memory.
            - Uploads the CSV to the S3 bucket at the path: project/version/stage/filename.
            - Logs the upload progress and success.

        Raises:
            Any exceptions from the S3 client (e.g., connection issues, permissions) are propagated to the caller.
        """

        key = f"{project}/{version}/{stage}/{filename}"
        self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Uploading '{filename}' ({len(df)} rows)")

        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        self.s3_client.put_object(
            Bucket=self.S3_BUCKET_NAME,
            Key=key,
            Body=buffer.getvalue()
        )

        self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Upload successful")
    
    
    def upload_default_training_config(self, model_name: str, version: str):
        """
        Upload the default training configuration file for a model version.

        Loads the default training YAML template, injects version-specific
        values (such as the model output directory), and uploads the finalized
        configuration to the versioned training configuration path in storage.

        This ensures every model version starts with a consistent and
        reproducible training configuration.

        Args:
            model_name (str): Name of the model (e.g., "ladybug").
            version (str): Version identifier (e.g., "v1", "v2").

        Raises:
            FileNotFoundError: If the default training template is missing.
            Exception: Propagates any storage upload failures.
        """
        template_path = Path("/app/ladybug/assets/training_defaults/default_training.yaml")

        self.logger.info(
            f"[{PipelineStep.STORAGE}] [{StorageType.S3}] "
            f"Preparing default training config for model='{model_name}', version='{version}'"
        )

        # Validate template exists
        if not template_path.exists():
            raise FileNotFoundError(
                f"Default training template not found at '{template_path}'. "
                "Ensure the assets directory is available in the runtime environment."
            )

        # Read and render template
        text = template_path.read_text()
        text = text.format(
            output_dir=f"{model_name}/{version}/model"
        )

        # Write to temporary file
        tmp_file = "/tmp/default_training_config.yaml"
        Path(tmp_file).write_text(text)

        s3_key = f"{model_name}/{version}/training/training_configs/config.yaml"

        # Upload to storage
        self.upload(local_file_path=tmp_file, s3_key=s3_key)

        self.logger.info(
            f"[{PipelineStep.STORAGE}] [{StorageType.S3}] "
            f"Uploaded default training config to '{s3_key}'"
        )


    
    def create_version_folders(self, model_name: str, version: str):
        """
        Initialize the storage folder structure for a specific model version.

        Creates all required directories for the ML pipeline lifecycle
        (raw data, processed data, training artifacts, models, logs, etc.)
        under a versioned prefix in object storage.

        Folder creation is done by uploading a lightweight placeholder file
        ('.keep') to each directory path.

        Also ensures that a shared 'production' directory exists for the model.

        Args:
            model_name (str): Name of the model (e.g., "ladybug").
            version (str): Version identifier (e.g., "v1", "v2").

        Raises:
            Exception: Propagates any exception raised during storage operations.
        """
        version_prefix = f"{model_name}/{version}/"

        folders = [
            "raw_data/",
            "processed_data/",
            "training/",
            "training/training_configs/",
            "training/training_data/",
            "mlflow/",
            "mlflow/artifacts/",
            "model/",
            "logs/",
        ]

        self.logger.info(
            f"[{PipelineStep.STORAGE}] [{StorageType.S3}] "
            f"Initializing storage structure for model='{model_name}', version='{version}'"
        )

        # Create version-specific folders
        for folder in folders:
            key = f"{version_prefix}{folder}.keep"

            self.upload(local_file_path=__file__, s3_key=key)

            self.logger.debug(
                f"[{PipelineStep.STORAGE}] [{StorageType.S3}] "
                f"Created folder placeholder '{key}'"
            )

        # Ensure production folder exists (shared across versions)
        prod_key = f"{model_name}/production/.keep"
        self.upload(local_file_path=__file__, s3_key=prod_key)

        self.logger.info(
            f"[{PipelineStep.STORAGE}] [{StorageType.S3}] "
            f"Storage structure ready for model='{model_name}', version='{version}' "
            f"(production folder ensured)"
        )

    
    def upload_json(self, data: dict, project: str, version: str, stage: str, filename: str):
        key = f"{project}/{version}/{stage}/{filename}"

        try:
            json_body = json.dumps(data, ensure_ascii=False, indent=2)

            self.s3_client.put_object(
                Bucket=self.S3_BUCKET_NAME,
                Key=key,
                Body=json_body,
                ContentType="application/json"
            )

            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Uploaded {filename} to S3 at {key}")
        except Exception as e:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Failed to upload at {key}: {e}")
            raise
        
    
    def upload_yaml(self, data: dict, project: str, version: str, stage: str, filename: str):
        key = f"{project}/{version}/{stage}/{filename}"

        try:
            yaml_body = yaml.dump(data, allow_unicode=True)

            self.s3_client.put_object(
                Bucket=self.S3_BUCKET_NAME,
                Key=key,
                Body=yaml_body,
                ContentType="application/x-yaml"
            )

            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Uploaded {filename} to S3 at {key}")
            return key

        except Exception as e:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Failed to upload at {key}: {e}")
            raise

    # ---------------- LOAD METHODS ----------------
    
    def get_df(self, project, version, stage, filename):
        key = f"{project}/{version}/{stage}/{filename}"
        
        try:
            obj = self.s3_client.get_object(
                Bucket=self.S3_BUCKET_NAME,
                Key=key
            )

            # read CSV from S3 into DF
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            return df

        except Exception as e:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Failed to fetch dataframe from S3 at {key}: {e}")
            raise Exception(f"Failed to fetch dataframe from S3 at {key}: {e}")



    def download_training_folder(self, project: str, version: str, local_base_dir: str):
        """
        Download the entire `training/` folder from S3 into a local directory.

        S3:
            project/version/training/...

        Local:
            local_base_dir/project/version/training/...
        """

        from pathlib import Path

        s3_prefix = f"{project}/{version}/training/"
        local_root = Path(local_base_dir) / project / version / "training"
        local_root.mkdir(parents=True, exist_ok=True)

        paginator = self.s3_client.get_paginator("list_objects_v2")
        found_files = False

        try:
            for page in paginator.paginate(
                Bucket=self.S3_BUCKET_NAME,
                Prefix=s3_prefix
            ):
                contents = page.get("Contents", [])
                if not contents:
                    continue

                for obj in contents:
                    key = obj["Key"]

                    # Skip folders and placeholders
                    if key.endswith("/") or key.endswith(".keep"):
                        continue

                    found_files = True

                    # Build local path
                    relative_path = key[len(s3_prefix):]  # safer than replace()
                    local_path = local_root / relative_path

                    # Ensure directory exists
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    self.s3_client.download_file(
                        Bucket=self.S3_BUCKET_NAME,
                        Key=key,
                        Filename=str(local_path)
                    )

            if not found_files:
                self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] No files found under S3 prefix: {s3_prefix}")
            else:
                self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Training folder downloaded successfully")
            
            return str(local_root)

        except Exception as e:
            self.logger.info(f"[{PipelineStep.STORAGE}] [{StorageType.S3}] Failed to fetch Training folder from S3 at {key}: {e}")
            raise






    # def test_connection(self):
    #     try:
    #         self.s3_client.list_objects_v2(Bucket=self.S3_BUCKET_NAME, MaxKeys=1)
    #         return True, f"Successfully connected to bucket: {self.S3_BUCKET_NAME}"
    #     except NoCredentialsError:
    #         return False, "AWS credentials are invalid or not provided."
    #     except ClientError as e:
    #         return False, f"Failed to connect to bucket: {e}"
    

    
    def upload(self, local_file_path: str, s3_key: str = None):
        file_path = Path(local_file_path)

        if not file_path.is_absolute():
            return {
                s3_key or local_file_path: [
                    False,
                    f"Expected absolute path, got relative: {local_file_path}"
                ]
            }

        if not file_path.exists():
            return {
                s3_key or file_path.name: [
                    False,
                    f"Local file not found: {file_path}"
                ]
            }

        if s3_key is None:
            s3_key = file_path.name

        try:
            self.s3_client.upload_file(
                Filename=str(file_path),
                Bucket=self.S3_BUCKET_NAME,
                Key=s3_key
            )
            return {s3_key: [True, "File uploaded successfully"]}
        except Exception as e:
            return {s3_key: [False, f"Upload failed: {e}"]}