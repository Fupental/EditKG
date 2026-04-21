import os
from pathlib import Path


MODELSCOPE_CACHE_ROOT = Path(
    os.environ.get("MODELSCOPE_CACHE", "/root/autodl-tmp/data/models/modelscope")
).expanduser()
QWEN3_MODEL_RELATIVE_PATH = Path("hub") / "models" / "Qwen" / "Qwen3-4B-Instruct-2507"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parent


def datasets_root() -> Path:
    local_path = project_root() / "data" / "datasets"
    if local_path.exists():
        return local_path
    return Path("/root/autodl-tmp/data/datasets")


def models_root() -> Path:
    local_path = project_root() / "data" / "models"
    if local_path.exists():
        return local_path
    return Path("/root/autodl-tmp/data/models")


def default_dataset_root() -> str:
    return str(datasets_root())


def default_adapter_path() -> str:
    return str(models_root() / "checkpoint-37736")


def default_base_model_path() -> str:
    return str(MODELSCOPE_CACHE_ROOT / QWEN3_MODEL_RELATIVE_PATH)


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
