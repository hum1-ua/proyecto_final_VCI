import os
import shutil
import numpy as np

IMAGES_PATH = "data/images"
MASKS_PATH = "data/masks"  
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"


def copy_split_files(file_list: list[str], dest_dir: str) -> None:
    """Copies images (and masks if available) into dest_dir/images and dest_dir/masks."""
    img_dest = os.path.join(dest_dir, "images")
    os.makedirs(img_dest, exist_ok=True)

    has_masks = os.path.isdir(MASKS_PATH)
    if has_masks:
        mask_dest = os.path.join(dest_dir, "masks")
        os.makedirs(mask_dest, exist_ok=True)

    for fname in file_list:
        shutil.copy2(os.path.join(IMAGES_PATH, fname), os.path.join(img_dest, fname))
        if has_masks and os.path.isfile(os.path.join(MASKS_PATH, fname)):
            shutil.copy2(os.path.join(MASKS_PATH, fname), os.path.join(mask_dest, fname))


def stratified_split(test_ratio: float = 0.3, seed: int = 42) -> None:
    """Creates stratified train and test folders preserving green/red orchard proportions."""
    rng = np.random.default_rng(seed)
    files = sorted(f for f in os.listdir(IMAGES_PATH) if f.endswith(".png"))

    green_files = [f for f in files if int(f.split("_")[1].split(".")[0]) < 66]
    red_files = [f for f in files if int(f.split("_")[1].split(".")[0]) >= 66]

    rng.shuffle(green_files)
    rng.shuffle(red_files)

    n_green_test = int(round(len(green_files) * test_ratio))
    n_red_test = int(round(len(red_files) * test_ratio))

    test_files = sorted(green_files[:n_green_test] + red_files[:n_red_test])
    train_files = sorted(green_files[n_green_test:] + red_files[n_red_test:])

    # Clean previous splits if re-running
    for folder in (TRAIN_DIR, TEST_DIR):
        if os.path.exists(folder):
            shutil.rmtree(folder)

    copy_split_files(train_files, TRAIN_DIR)
    copy_split_files(test_files, TEST_DIR)

    print(f"Train set ({len(train_files)} imgs): {len(green_files) - n_green_test} green, {len(red_files) - n_red_test} red")
    print(f"Test set  ({len(test_files)} imgs): {n_green_test} green, {n_red_test} red")


if __name__ == "__main__":
    stratified_split(test_ratio=0.3, seed=42)