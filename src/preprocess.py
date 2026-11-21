import os
import json
import cv2
import numpy as np

IMAGES_PATH = "data/images"
OUTPUT_PATH = "data/preprocessed"
SPLIT_PATH = "data/split.json"


def stratified_split(
    images_dir: str, test_ratio: float = 0.3, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Splits image filenames into stratified train/test lists."""
    rng = np.random.default_rng(seed)
    files = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))

    green_files = [f for f in files if int(f.split("_")[1].split(".")[0]) < 66]
    red_files = [f for f in files if int(f.split("_")[1].split(".")[0]) >= 66]

    rng.shuffle(green_files)
    rng.shuffle(red_files)

    n_green_test = int(round(len(green_files) * test_ratio))
    n_red_test = int(round(len(red_files) * test_ratio))

    test_files = sorted(green_files[:n_green_test] + red_files[:n_red_test])
    train_files = sorted(green_files[n_green_test:] + red_files[n_red_test:])
    return train_files, test_files


def is_red_apple_tree(
    img_lab_smooth: np.ndarray, a_thresh: float = 142.0, min_blob_area: int = 150
) -> bool:
    """Checks for a foreground red blob that is not dwarfed by yellow-green apples."""
    a = img_lab_smooth[:, :, 1]
    b = img_lab_smooth[:, :, 2]

    # 1. Require at least one pure-red connected component (filters bark and distant dots)
    red_candidates = (a > a_thresh).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(red_candidates)
    if num_labels <= 1 or np.max(stats[1:, cv2.CC_STAT_AREA]) < min_blob_area:
        return False

    # 2. Ensure red fruit area is not a minor background artifact in a green orchard
    red_area = np.sum(a > 138.0)
    green_apple_area = np.sum(b > 185.0)
    return bool(red_area > 0.20 * green_apple_area)


def preprocess_array(img_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Converts BGR to LAB, smooths, selects a* or b* channel, and normalizes to [0, 255]."""
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_smooth = cv2.bilateralFilter(img_lab, d=9, sigmaColor=25, sigmaSpace=50)

    red_tree = is_red_apple_tree(img_smooth)

    if red_tree:
        # Use a* channel: suppresses grass and green background apples
        a = img_smooth[:, :, 1].astype(np.float32)
        out = np.clip((a - 126.0) / (155.0 - 126.0), 0.0, 1.0) * 255.0
    else:
        # Use b* channel: isolates yellow-green foreground apples
        b = img_smooth[:, :, 2].astype(np.float32)
        out = np.clip((b - 152.0) / (188.0 - 152.0), 0.0, 1.0) * 255.0

    return out.astype(np.uint8), red_tree


def preprocess_image(img_path: str) -> np.ndarray:
    """Reads an image from disk and returns its single-channel preprocessed map."""
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    processed, _ = preprocess_array(img_bgr)
    return processed


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    train_files, test_files = stratified_split(IMAGES_PATH, test_ratio=0.3, seed=42)
    with open(SPLIT_PATH, "w") as f:
        json.dump({"train_files": train_files, "test_files": test_files}, f, indent=2)

    errors = []
    for img_name in sorted(os.listdir(IMAGES_PATH)):
        if not img_name.endswith(".png"):
            continue

        in_path = os.path.join(IMAGES_PATH, img_name)
        img_bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
        processed, pred_red = preprocess_array(img_bgr)
        cv2.imwrite(os.path.join(OUTPUT_PATH, img_name), processed)

        img_id = int(img_name.split("_")[1].split(".")[0])
        true_red = img_id >= 66
        if pred_red != true_red:
            errors.append((img_name, pred_red))

    print(f"Split saved -> Train: {len(train_files)} | Test: {len(test_files)}")
    print(f"Misclassified tree types: {len(errors)} -> {errors}")