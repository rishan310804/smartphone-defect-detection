import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Project paths (should not need to be changed)
PROJECT_DIR = Path(__file__).resolve().parent
RAW_IMAGES_DIR = PROJECT_DIR / "data" / "raw" / "images"
PROCESSED_LABELS_DIR = PROJECT_DIR / "data" / "processed" / "yolo_labels"
SPLITS_DIR = PROJECT_DIR / "data" / "splits"
CONFIGS_DIR = PROJECT_DIR / "configs"

# Defect classes
CLASSES = [
    'crack', 'broken', 'spot', 'scratch',
    'light-leakage', 'blot', 'broken-membrane'
]


def prepare_dataset():
    """
    Creates dataset.yaml and splits the data into train/val/test sets.
    """
    # 1. Ensure all directories exist
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Skip image moving step
    print("Images are already in place.")

    # 3. Create dataset.yaml configuration file
    print("Creating dataset.yaml...")
    config = {
        'path': str(SPLITS_DIR.parent.absolute()),
        'train': 'splits/train/images',
        'val': 'splits/val/images',
        'test': 'splits/test/images',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    config_path = CONFIGS_DIR / "dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Created dataset config: {config_path}")

    # 4. Split the dataset
    print("Splitting dataset...")
    images = list(RAW_IMAGES_DIR.glob("*.jpg"))
    labels = list(PROCESSED_LABELS_DIR.glob("*.txt"))

    # Ensure we only use images that have corresponding labels
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    valid_stems = image_stems.intersection(label_stems)

    valid_images = [p for p in images if p.stem in valid_stems]

    if not valid_images:
        print("ERROR: No images with corresponding labels were found. Make sure images are in 'data/raw/images' and labels are in 'data/processed/yolo_labels'.")
        return

    train_val, test_images = train_test_split(
        valid_images, test_size=0.1, random_state=42)
    train_images, val_images = train_test_split(
        train_val, test_size=0.22, random_state=42)  # 0.22 * 0.9 = approx 0.2

    splits = {'train': train_images, 'val': val_images, 'test': test_images}

    for split_name, image_list in splits.items():
        img_dir = SPLITS_DIR / split_name / "images"
        lbl_dir = SPLITS_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_list:
            # Copy image
            shutil.copy2(img_path, img_dir / img_path.name)
            # Copy corresponding label
            label_path = PROCESSED_LABELS_DIR / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, lbl_dir / label_path.name)

    print("Dataset split complete:")
    print(f"    Train: {len(train_images)} images")
    print(f"    Validation: {len(val_images)} images")
    print(f"    Test: {len(test_images)} images")


if __name__ == "__main__":
    prepare_dataset()
