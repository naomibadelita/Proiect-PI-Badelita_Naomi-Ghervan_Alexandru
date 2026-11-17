from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 8: 5}

src_root = PROJECT_ROOT / "dataset" / "raw" / "labels_backup"
dst_root = PROJECT_ROOT / "dataset" / "raw" / "labels"

for split in ["train", "val", "test"]:
    src = src_root / split
    dst = dst_root / split
    if not src.exists():
        print(f"Lipsește: {src}")
        continue
    dst.mkdir(parents=True, exist_ok=True)
    for txt in src.glob("*.txt"):
        new_lines = []
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            if old_id in CLASS_MAP:
                parts[0] = str(CLASS_MAP[old_id])
                new_lines.append(" ".join(parts))
        (dst / txt.name).write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    print(f"{split}: gata")