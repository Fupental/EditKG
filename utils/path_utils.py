import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parent


def datasets_root() -> Path:
    return project_root() / "data" / "datasets"


def models_root() -> Path:
    return project_root() / "data" / "models"


def default_dataset_root() -> str:
    return str(datasets_root())


def default_adapter_path() -> str:
    return str(models_root() / "checkpoint-37736")


def default_base_model_path() -> str:
    return str(Path("/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"))


def resolve_data_path(data_path: str) -> str:
    path = Path(data_path).expanduser()
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return str(path)


def resolve_dataset_dir(data_path: str, dataset: str) -> str:
    return str(Path(resolve_data_path(data_path)) / dataset)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
