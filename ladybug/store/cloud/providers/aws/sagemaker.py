import boto3
from botocore.exceptions import ClientError, NoCredentialsError

class SageMaker():
    def __init__(self, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION):
        self.AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID
        self.AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY
        self.AWS_REGION = AWS_REGION

        # Initialize SageMaker client
        self.sagemaker_client = boto3.client(
            "sagemaker",
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_REGION
        )

    def test_connection(self):
        """
        Simple health check:
        tries listing 1 training job.
        If you have zero jobs, AWS still returns success.
        """
        try:
            self.sagemaker_client.list_training_jobs(MaxResults=1)
            return True, "Successfully connected to SageMaker"
        
        except NoCredentialsError:
            return False, "AWS credentials are invalid or not provided."
        
        except ClientError as e:
            return False, f"Failed to connect to SageMaker: {e}"

        except Exception as e:
            return False, f"Unexpected error: {e}"
