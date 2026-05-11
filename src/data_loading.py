from pathlib import Path
import pandas as pd

def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src").is_dir() and (candidate / "data" / "students_dataset.csv").is_file():
            return candidate
    raise RuntimeError("Could not locate repository root containing src/ and data/students_dataset.csv")


def load_students_dataset(repo_root: Path, filename: str = "students_dataset.csv") -> pd.DataFrame:
    data_path = repo_root / "data" / filename
    return pd.read_csv(data_path)
