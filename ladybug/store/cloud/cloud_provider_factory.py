from ladybug.core.config import Settings
from .cloud_enum import StorageEnum, TrainingEnum
from .providers.aws.sagemaker import SageMaker
from .providers.aws.s3 import S3

class CloudProviderFactory:
    def __init__(self, config: Settings):
        self.config = config
    
    def create_storage(self, provider: str):

        if provider == StorageEnum.S3.value:
            return S3(
                AWS_ACCESS_KEY_ID=self.config.AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY=self.config.AWS_SECRET_ACCESS_KEY,
                AWS_REGION=self.config.AWS_REGION,
                S3_BUCKET_NAME=self.config.S3_BUCKET_NAME
            )
        
        return None
    
    def create_training(self, provider: str):

        if provider == TrainingEnum.SAGEMAKER.value:
            return SageMaker(
                AWS_ACCESS_KEY_ID=self.config.AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY=self.config.AWS_SECRET_ACCESS_KEY,
                AWS_REGION=self.config.AWS_REGION,
            )
        
        return None