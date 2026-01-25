import random
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

# Find the images directory - Chest X-Ray dataset is organized in train/test/val folders
# Each folder contains NORMAL and PNEUMONIA subfolders
source_base_dir = None
for possible_dir in [
    Path(path) / "chest_xray",
    Path(path) / "Chest X-Ray",
    Path(path) / "data" / "chest_xray",
    Path(path),
]:
    if possible_dir.exists() and possible_dir.is_dir():
        # Check if it contains train/test/val folders
        has_train = (possible_dir / "train").exists() or (possible_dir / "Train").exists()
        if has_train:
            source_base_dir = possible_dir
            break

if source_base_dir is None:
    # Try to find by searching for common folder names
    for candidate_dir in Path(path).rglob("train"):
        if candidate_dir.is_dir():
            source_base_dir = candidate_dir.parent
            break

if source_base_dir is None:
    raise FileNotFoundError(f"Could not find chest X-ray dataset structure in {path}")

print(f"Source base directory: {source_base_dir}")

# Find train, test, or val directories (prefer train, then test, then val)
data_dirs = []
for split_name in ["train", "Train", "test", "Test", "val", "Val"]:
    split_dir = source_base_dir / split_name
    if split_dir.exists() and split_dir.is_dir():
        data_dirs.append(split_dir)
        break

if len(data_dirs) == 0:
    # Try to find any directory with class subfolders
    for candidate_dir in source_base_dir.iterdir():
        if candidate_dir.is_dir():
            subdirs = [d for d in candidate_dir.iterdir() if d.is_dir()]
            class_names = [d.name for d in subdirs]
            if "NORMAL" in class_names or "PNEUMONIA" in class_names or "normal" in [c.lower() for c in class_names]:
                data_dirs.append(candidate_dir)
                break

if len(data_dirs) == 0:
    raise FileNotFoundError(
        f"Could not find train/test/val directory with NORMAL/PNEUMONIA subfolders in {source_base_dir}"
    )

source_data_dir = data_dirs[0]
print(f"Using data directory: {source_data_dir}")

# Get all class directories (NORMAL and PNEUMONIA)
class_dirs = [d for d in source_data_dir.iterdir() if d.is_dir()]
class_dirs = [d for d in class_dirs if d.name.upper() in ["NORMAL", "PNEUMONIA"]]

if len(class_dirs) == 0:
    # Try case-insensitive search
    for d in source_data_dir.iterdir():
        if d.is_dir() and d.name.upper() in ["NORMAL", "PNEUMONIA"]:
            class_dirs.append(d)

if len(class_dirs) == 0:
    raise FileNotFoundError(f"Could not find NORMAL and PNEUMONIA class directories in {source_data_dir}")

print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

# Sample images from each class
random.seed(42)
total_samples = 10000
samples_per_class = total_samples // len(class_dirs)
remaining_samples = total_samples % len(class_dirs)

print(f"Sampling {samples_per_class} images per class (with {remaining_samples} extra samples)")

# Create target directories
target_dir = Path(f"tests/data/chestxray-dataset-{total_samples}")
target_dir.mkdir(parents=True, exist_ok=True)
images_dir = target_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Collect all image data
image_data = []
copied_count = 0
image_counter = 0

# Common image extensions
image_extensions = [".jpg", ".jpeg", ".png"]

for class_dir in class_dirs:
    class_name = class_dir.name

    # Get all image files in this class directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(class_dir.glob(f"*{ext}")))
        image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))

    if len(image_files) == 0:
        print(f"Warning: No images found in class directory {class_dir}")
        continue

    # Determine how many samples to take from this class
    num_samples = samples_per_class
    if remaining_samples > 0:
        num_samples += 1
        remaining_samples -= 1

    # Sample images (with replacement if the class is too small)
    if num_samples <= len(image_files):
        sampled_images = random.sample(image_files, num_samples)
        print(f"Class {class_name}: sampling {num_samples} from {len(image_files)} images")
    else:
        # Take all unique images, then top up by sampling with replacement.
        sampled_images = list(image_files)
        extra = num_samples - len(image_files)
        sampled_images.extend(random.choices(image_files, k=extra))
        print(
            f"Class {class_name}: sampling {num_samples} from {len(image_files)} images "
            f"(with replacement for {extra})"
        )

    # Copy images and record metadata
    for image_file in sampled_images:
        # Get the original file extension
        original_ext = image_file.suffix.lower()
        if not original_ext:
            original_ext = ".jpg"

        # Generate generic filename that doesn't reveal the class
        image_counter += 1
        image_filename = f"image_{image_counter:04d}{original_ext}"
        dest_image = images_dir / image_filename

        shutil.copy2(image_file, dest_image)
        copied_count += 1

        # Record image metadata
        image_data.append(
            {
                "filename": image_filename,
                "path": f"{target_dir}/images/{image_filename}",
                "class": class_name,
            }
        )

print(f"Copied {copied_count} images to {images_dir}")

# Create DataFrame and save CSV
df_subsample = pd.DataFrame(image_data)
output_path = target_dir / "chestxray.csv"
df_subsample.to_csv(output_path, index=False)
print(f"Saved subsampled CSV to {output_path}")
print(f"Subsample contains {len(df_subsample)} rows with {copied_count} images")
print("Class distribution:")
print(df_subsample["class"].value_counts().sort_index())
