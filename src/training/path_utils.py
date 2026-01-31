from pathlib import Path


def resolve_data_dir(default_path: str = "../data/processed") -> str:
    """Resolve data directory, preferring SageMaker NVMe storage when available."""
    nvme_path = Path("/mnt/sagemaker-nvme/data/processed")
    if nvme_path.exists():
        return str(nvme_path)
    return default_path
