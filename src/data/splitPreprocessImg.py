import os
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

from src.preprocess import enhance   

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Preprocessor")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATASET_DIR = (PROJECT_ROOT / os.getenv("DATASET_DIR", "./dataset")).resolve()
SRC_DIR = DATASET_DIR / "raw"
DEST_DIR = DATASET_DIR / "preprocessed"

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
#SPLITS = ["train", "val", "test"]
SPLITS = ["val"] 

def preprocess_dataset(src_dir: Path, dest_dir: Path):
    for split in SPLITS:
        src_img_dir = src_dir / "images" / split
        src_lbl_dir = src_dir / "labels" / split
        dst_img_dir = dest_dir / "images" / split
        dst_lbl_dir = dest_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in src_img_dir.iterdir()
                if f.suffix.lower() in VALID_IMAGE_EXTS]
        logger.info("Split %s: %d images", split, len(files))

        ok, failed = 0, 0
        for img_path in files:
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    out = enhance(im)
                    out.save(dst_img_dir / img_path.name)

                lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
                ok += 1
            except Exception as e:
                failed += 1
                logger.warning("Failure at %s: %s", img_path.name, e)

        logger.info("Split %s done: %d ok, %d failed", split, ok, failed)


if __name__ == "__main__":
    preprocess_dataset(SRC_DIR, DEST_DIR)