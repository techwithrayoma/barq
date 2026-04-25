from enum import StrEnum

# ---------------- Pipeline Steps ----------------
class PipelineStep(StrEnum):
    PIPELINE = "PIPELINE"
    CREATE_FOLDERS = "CREATE_FOLDERS"
    INGESTION = "INGESTION"
    CLEAN = "CLEAN"
    LABELING = "LABELING"
    TRANSFORMATION = "TRANSFORMATION"
    TRAINING = "TRAINING"
    STORAGE = "STORAGE"


# ---------------- Storage Types ----------------
class StorageType(StrEnum):
    LOCAL = "LOCAL"
    S3 = "S3"