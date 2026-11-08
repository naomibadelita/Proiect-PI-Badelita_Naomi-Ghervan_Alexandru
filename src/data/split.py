import os
import shutil
import random
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DatasetSplitter")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

RAW_DIR_ENV = os.getenv("DATASET_RAW_DIR", "./data_raw")
DEST_DIR_ENV = os.getenv("DATASET_DIR", "./dataset")

RAW_DIR = (PROJECT_ROOT / RAW_DIR_ENV).resolve()
DEST_DIR = (PROJECT_ROOT / DEST_DIR_ENV).resolve() / "raw"

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def split_dataset(source_dir: Path, dest_dir: Path, train_ratio: float=0.8,val_ratio: float=0.1,test_ratio: float=0.1, seed: int = 42):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("Suma train_ratio + val_ratio + test_ratio trebuie sa fie egala cu 1.0!")

    logger.info("Initialization split process (Train: %.0f%%, Val: %.0f%%, Test: %.0f%%)...",train_ratio * 100, val_ratio * 100, test_ratio * 100)
    logger.info("Source: %s", source_dir)
    logger.info("Destination: %s", dest_dir)

    src_images = source_dir / "images"
    src_labels = source_dir / "labels"

    splits = ["train", "val", "test"]
    subsets = {}

    for split in splits:
        subsets[split] = {
            "img_dir": dest_dir / "images" / split,
            "lbl_dir": dest_dir / "labels" / split
        }

    for sub in subsets.values():
        sub["img_dir"].mkdir(parents=True, exist_ok=True)
        sub["lbl_dir"].mkdir(parents=True, exist_ok=True)

    image_files=[]
    for file in src_images.iterdir():
        if file.suffix.lower() in VALID_IMAGE_EXTS:
            image_files.append(file)

    valid_pairs = []
    for img_path in image_files:
        lbl_path = src_labels / f"{img_path.stem}.txt"
        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))

    total_pairs = len(valid_pairs)
    if total_pairs == 0:
        logger.error("Couldn't find valid pairs (image, label) to process in %s!", source_dir)
        return

    random.seed(seed)
    random.shuffle(valid_pairs)

    train_end = int(total_pairs * train_ratio)
    val_end = train_end + int(total_pairs * val_ratio)

    partitions = {
        "train": valid_pairs[:train_end],
        "val": valid_pairs[train_end:val_end],
        "test": valid_pairs[val_end:]
    }

    for split_name, pairs in partitions.items():
        logger.info("Copying subset %s (%d pairs)...", split_name.capitalize(), len(pairs))
        target_img_dir = subsets[split_name]["img_dir"]
        target_lbl_dir = subsets[split_name]["lbl_dir"]

        for img, lbl in pairs:
            shutil.copy2(img, target_img_dir / img.name)
            shutil.copy2(lbl, target_lbl_dir / lbl.name)

    logger.info("Split finalized successfully!")
    logger.info(
        "Total: %d pairs | Train: %d (%.1f%%) | Val: %d (%.1f%%) | Test: %d (%.1f%%)",
        total_pairs,
        len(partitions["train"]), (len(partitions["train"]) / total_pairs) * 100,
        len(partitions["val"]), (len(partitions["val"]) / total_pairs) * 100,
        len(partitions["test"]), (len(partitions["test"]) / total_pairs) * 100
    )

if __name__ == "__main__":
    split_dataset(
        source_dir=RAW_DIR,
        dest_dir=DEST_DIR,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )