from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import torch as t


def repo_path_to_abs_path(path: str) -> Path:
    repo_abs_path = Path(__file__).parent.parent.absolute()
    return repo_abs_path / path


def save_cache(data_dict: Dict[Any, Any], folder_name: str, base_filename: str):
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = folder / f"{base_filename}-{dt_string}.pkl"
    print(f"Saving cache to {file_path}")
    t.save(data_dict, file_path)


def load_cache(folder_name: str, filename: str) -> Dict[Any, Any]:
    folder = repo_path_to_abs_path(folder_name)
    return t.load(folder / filename)

