import os
import logging
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger=logging.getLogger("DataVerifier")

PROJECT_ROOT=Path(__file__).resolve().parents[2]
#loads what is in the .env file in the os env variables
load_dotenv(PROJECT_ROOT / ".env") 

RAW_DIR_ENV=os.getenv("DATASET_RAW_DIR", "./data_raw")
RAW_DIR=(PROJECT_ROOT / RAW_DIR_ENV).resolve()

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def verify_raw_dataset(source_dir: Path):
    logger.info("Start verification of the raw dataset at the Path: %s",source_dir)
    images_dir=source_dir / "images"
    labels_dir=source_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        logger.error("Invalid structure! There must be an 'images' folder and an 'labels' folder in the %s", source_dir)
        return False

    images_files=[]
    for file in images_dir.iterdir():
        if file.suffix.lower() in VALID_IMAGE_EXTS:
            images_files.append(file)

    labels_files={}
    for file in labels_dir.iterdir():
        if file.suffix.lower()==".txt":
            labels_files[file.stem]=file

    logger.info("Number of found images: %d",len(images_files))
    logger.info("Number of found labels: %d",len(labels_files))

    missing_labels=0
    corrupt_images=0
    invalid_bboxes=0
    class_counter=Counter()

    for img_path in images_files:
        stem=img_path.stem
        img=cv2.imread(str(img_path))
        if img is None:
            logger.warning("Corrupt image, impossible to decode: %s",img_path.name)
            corrupt_images+=1
            continue

        if stem not in labels_files:
            missing_labels+=1
            continue

        label_path=labels_files[stem]
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue

        if len(parts) != 5:
                logger.warning(
                    "Invalid format %s at line %d: expected 5 elements, found %d",
                    label_path.name, line_idx + 1, len(parts)
                )
                invalid_bboxes += 1
                continue
        
        class_id, x_center, y_center, width, height = parts

        try:
            class_id = int(class_id)
            coords = [float(x_center), float(y_center), float(width), float(height)]
        except ValueError:
            logger.warning("Numeric values in %s: %s", label_path.name, line.strip())
            invalid_bboxes += 1
            continue

        if not all(0.0 <= c <= 1.0 for c in coords):
            logger.warning(
                "Coordinates outside the interval [0, 1] in %s: %s",
                label_path.name, coords
            )
            invalid_bboxes += 1
            continue

        class_counter[class_id] += 1

    image_stems=set()
    for file in images_files:
        image_stems.add(file.stem)

    orphan_labels=[]
    for stem in labels_files:
        if stem not in image_stems:
            orphan_labels.append(stem)

    # Format the class distribution breakdown into sorted lines separated by newlines
    class_lines = "\n".join(
        f"  Class {cls_id}: {class_counter[cls_id]} instances"
        for cls_id in sorted(class_counter.keys())
    )

    report = (
        "\n" + "=" * 50 + "\n"
        "            DATA INTEGRITY REPORT            \n"
        + "=" * 50 + "\n"
        
        f"Total scanned images:         {len(images_files)}\n"
        f"Corrupt images:               {corrupt_images}\n"
        f"Images missing labels:        {missing_labels}\n"
        f"Orphan labels (no image):     {len(orphan_labels)}\n"
        f"Invalid bounding boxes:       {invalid_bboxes}\n"
        
        + "-" * 50 + "\n"
        "Class distribution (Class Balance):\n"
        f"{class_lines}\n"
        
        + "=" * 50
    )

    logger.info("%s", report)

if __name__ == "__main__":
    verify_raw_dataset(RAW_DIR)


