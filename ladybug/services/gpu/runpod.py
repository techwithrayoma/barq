class RunPodService:
    """
    Handles submitting ML training jobs to RunPod GPU instances.
    """

    def __init__(self, api_key: str, default_image: str = "my-ml-image:latest"):
        self.api_key = api_key
        self.default_image = default_image
        # You can store endpoints or configs here

    def submit_training_job(self, training_assets_path: str, config_file: str):
        """
        Sends a job to RunPod with your training data and config.
        """
        # Example pseudocode:
        payload = {
            "image": self.default_image,
            "command": f"python train.py --config {config_file}",
            "volumes": [training_assets_path],
            "gpu": True
        }

        response = self._call_runpod_api(payload)
        return response["job_id"]

    def check_job_status(self, job_id: str):
        """
        Poll the RunPod API to check job progress or completion.
        """
        response = self._call_runpod_api({"job_id": job_id}, endpoint="status")
        return response["status"]

    def download_outputs(self, job_id: str, local_path: str):
        """
        Pulls trained model and logs back to your storage.
        """
        # Download files using API
        self._call_runpod_api({"job_id": job_id, "destination": local_path}, endpoint="download")

    def _call_runpod_api(self, payload: dict, endpoint: str = "submit"):
        """
        Low-level API call wrapper
        """
        # requests.post(f"{RUNPOD_API}/{endpoint}", headers=..., json=payload)
        pass
