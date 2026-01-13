import os
import random
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("apollo2506/eurosat-dataset")

print("Path to dataset files:", path)

# Find the images directory - EuroSAT is typically organized by class folders
source_images_dir = None
for possible_dir in [
    Path(path) / "EuroSAT_RGB",
    Path(path) / "EuroSAT",
    Path(path) / "data" / "EuroSAT_RGB",
    Path(path) / "images",
    Path(path) / "data" / "images",
]:
    if possible_dir.exists() and possible_dir.is_dir():
        source_images_dir = possible_dir
        break

if source_images_dir is None:
    # Try to find images directory by searching for common EuroSAT folder names
    for img_dir in Path(path).rglob("EuroSAT*"):
        if img_dir.is_dir():
            source_images_dir = img_dir
            break

if source_images_dir is None:
    # Last resort: search for any directory that might contain class subdirectories
    for candidate_dir in Path(path).iterdir():
        if candidate_dir.is_dir():
            # Check if it contains subdirectories (likely class folders)
            subdirs = [d for d in candidate_dir.iterdir() if d.is_dir()]
            if len(subdirs) > 0:
                source_images_dir = candidate_dir
                break

if source_images_dir is None:
    raise FileNotFoundError(f"Could not find images directory in {path}")

print(f"Source images directory: {source_images_dir}")

# Get all class directories (subdirectories containing images)
class_dirs = [d for d in source_images_dir.iterdir() if d.is_dir()]
if len(class_dirs) == 0:
    # If no subdirectories, check if images are directly in the directory
    image_files = list(source_images_dir.glob("*.jpg")) + list(source_images_dir.glob("*.png"))
    if len(image_files) > 0:
        # Images are in a flat structure, we'll need to handle this differently
        # For now, assume class-based structure
        raise FileNotFoundError(
            f"Expected class-based directory structure, but found flat structure in {source_images_dir}"
        )

print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

# Sample images from each class
random.seed(42)  # For reproducibility
total_samples = 1000
samples_per_class = total_samples // len(class_dirs)
remaining_samples = total_samples % len(class_dirs)

print(f"Sampling {samples_per_class} images per class (with {remaining_samples} extra samples)")

# Create target directories
target_dir = Path(f"tests/data/eurosat-dataset-{total_samples}")
target_dir.mkdir(parents=True, exist_ok=True)
images_dir = target_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Collect all image data
image_data = []
copied_count = 0
image_counter = 0  # Counter for generating generic filenames

# Common image extensions
image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

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

    # Sample images
    num_samples = min(num_samples, len(image_files))
    sampled_images = random.sample(image_files, num_samples)

    print(f"Class {class_name}: sampling {num_samples} from {len(image_files)} images")

    # Copy images and record metadata
    for image_file in sampled_images:
        # Get the original file extension
        original_ext = image_file.suffix.lower()
        if not original_ext:
            # Default to .jpg if no extension found
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
output_path = target_dir / "eurosat.csv"
df_subsample.to_csv(output_path, index=False)
print(f"Saved subsampled CSV to {output_path}")
print(f"Subsample contains {len(df_subsample)} rows with {copied_count} images")
print("Class distribution:")
print(df_subsample["class"].value_counts().sort_index())
