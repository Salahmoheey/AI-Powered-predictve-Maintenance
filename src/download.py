"""Check for the NASA C-MAPSS FD001 files required by this project.

Dataset source: NASA Prognostics Center of Excellence (PCoE) Data Set Repository.
Landing page: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
The FD001 text files should be placed in data/raw/.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REQUIRED_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
DATASET_SOURCE = "NASA Prognostics Center of Excellence (PCoE) Data Set Repository"
DATASET_URL = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"


def main():
    """Report which raw dataset files are available and where to obtain them."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {DATASET_SOURCE}")
    print(f"Source: {DATASET_URL}")
    missing = []
    for file_name in REQUIRED_FILES:
        path = RAW_DIR / file_name
        if path.is_file() and path.stat().st_size > 0:
            print(f"Found: {path.name}")
        else:
            print(f"Missing: {path.name}")
            missing.append(file_name)
    if missing:
        print("Download the FD001 files from the source above and place them in:", RAW_DIR)
    else:
        print("All required FD001 files are ready.")


if __name__ == "__main__":
    main()
