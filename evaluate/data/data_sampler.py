import random
from pathlib import Path
from collections import defaultdict

# ---------------- CONFIG ----------------
INPUT_SPLIT = Path("eigen_train_files.txt")
OUTPUT_SPLIT = Path("eigen_train_files_20pct.txt")
IMAGES_PER_DRIVE = 200
RANDOM_SEED = 42
# ---------------------------------------

random.seed(RANDOM_SEED)

# Read split file
with INPUT_SPLIT.open() as f:
    lines = [line.strip() for line in f if line.strip()]

# Group by drive
drive_to_lines = defaultdict(list)

for line in lines:
    seq_path, frame_idx, cam = line.split()
    # "2011_09_26/2011_09_26_drive_0001_sync" → drive name
    drive = seq_path.split("/")[-1]
    drive_to_lines[drive].append(line)

# Sample per drive
selected_lines = []

for drive, drive_lines in sorted(drive_to_lines.items()):
    if len(drive_lines) < IMAGES_PER_DRIVE:
        continue  # skip short drives
    sampled = random.sample(drive_lines, IMAGES_PER_DRIVE)
    selected_lines.extend(sampled)

# Shuffle final list
random.shuffle(selected_lines)

# Write new split
with OUTPUT_SPLIT.open("w") as f:
    for line in selected_lines:
        f.write(line + "\n")

print(f"Saved {len(selected_lines)} samples to {OUTPUT_SPLIT}")
print(f"Used {len(selected_lines) // IMAGES_PER_DRIVE} drives")
