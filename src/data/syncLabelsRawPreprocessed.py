import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
src = PROJECT_ROOT / "dataset" / "raw" / "labels"
dst = PROJECT_ROOT / "dataset" / "preprocessed" / "labels"

for split in ["train", "val", "test"]:
    shutil.copytree(src / split, dst / split, dirs_exist_ok=True)
    print(f"{split}: done")