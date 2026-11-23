import os
import json
import cv2
import numpy as np

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
PARAMS_PATH = "data/preprocess_params.json"


def load_preprocess_params(params_path: str = PARAMS_PATH) -> float:
    """Loads the calibrated 99th percentile threshold for channel b* from JSON."""
    with open(params_path, "r") as f:
        params = json.load(f)
    return float(params.get("b_p99_thresh"))


def preprocess_array(img_bgr: np.ndarray, b_p99_thresh: float) -> tuple[np.ndarray, bool]:
    """Selects channel a* or b* using the calibrated P99 threshold and normalizes to [0, 255]."""
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_smooth = cv2.bilateralFilter(img_lab, d=9, sigmaColor=25, sigmaSpace=50)

    a = img_smooth[:, :, 1] #.astype(np.float32)
    b = img_smooth[:, :, 2] #.astype(np.float32)

    is_red_tree = bool(np.percentile(b, 99) < b_p99_thresh)

    if is_red_tree:
        # Red apple orchard: isolate red chromaticity in channel a*
        # out = np.clip((a - 126.0) / (155.0 - 126.0), 0.0, 1.0) * 255.0
        out = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    else:
        # Green apple orchard: isolate yellow-green chromaticity in channel b*
        # out = np.clip((b - 155.0) / (b_p99_thresh - 155.0), 0.0, 1.0) * 255.0
        out = cv2.normalize(b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return out.astype(np.uint8), is_red_tree

def preprocess_image(img_path: str, b_p99_thresh: float) -> np.ndarray:
    """Reads an image from disk and returns its single-channel preprocessed map."""
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    processed, _ = preprocess_array(img_bgr, b_p99_thresh)
    return processed


def process_split(split_dir: str, b_p99_thresh: float) -> None:
    """Preprocesses all images in split_dir/images and saves them to split_dir/preprocessed."""
    in_dir = os.path.join(split_dir, "images")
    out_dir = os.path.join(split_dir, "preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    errors = []
    count = 0

    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith(".png"):
            continue

        img_bgr = cv2.imread(os.path.join(in_dir, fname), cv2.IMREAD_COLOR)
        processed, pred_red = preprocess_array(img_bgr, b_p99_thresh)
        cv2.imwrite(os.path.join(out_dir, fname), processed)
        count += 1

        img_id = int(fname.split("_")[1].split(".")[0])
        true_red = img_id >= 66
        if pred_red != true_red:
            errors.append((fname, "pred_red" if pred_red else "pred_green"))

    print(f"[{split_dir}] Processed {count} images | Channel mismatches: {len(errors)} -> {errors}")


if __name__ == "__main__":
    b_thresh = load_preprocess_params(PARAMS_PATH)
    print(f"Using calibrated threshold b* (P99) = {b_thresh:.2f}\n")

    process_split(TRAIN_DIR, b_thresh)
    process_split(TEST_DIR, b_thresh)