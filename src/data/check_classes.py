from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
c = Counter()
for txt in (PROJECT_ROOT / "dataset" / "raw" / "labels" / "train").glob("*.txt"):
    for line in txt.read_text().splitlines():
        if line.strip():
            c[int(line.split()[0])] += 1
print(sorted(c.items()))