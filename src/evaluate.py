from sklearn.metrics import jaccard_score, precision_score, recall_score
import os
from typing import Callable
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

from segment_functions import segment_otsu, segment_watershed

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
PREPROC_SUBDIR = "preprocessed"
GT_SUBDIR = "masks"

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    y_pred = (pred_mask > 0).flatten()
    y_true = (gt_mask > 0).flatten()

    if not np.any(y_true) and not np.any(y_pred):
        return {"iou": 1.0, "precision": 1.0, "recall": 1.0}

    return {
        "iou": float(jaccard_score(y_true, y_pred, zero_division=0.0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
    }

def evaluate_method(
    split_dir: str,
    segment_fn: Callable[[np.ndarray], np.ndarray],
    method_name: str,
) -> dict[str, float]:
    """Evalúa cualquier función de segmentación sobre un split completo."""
    prep_dir = os.path.join(split_dir, PREPROC_SUBDIR)
    gt_dir = os.path.join(split_dir, GT_SUBDIR)

    red_ious, green_ious = [], []
    all_ious, all_prec, all_rec = [], [], []

    for fname in sorted(os.listdir(prep_dir)):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(prep_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None or gt_mask is None:
            raise FileNotFoundError(f"Falta imagen o máscara para: {fname}")

        pred_mask = segment_fn(img_gray)

        # Crea la carpeta y subcarpetas si no existen
        ruta = Path(f"results/{method_name}/{fname}")
        ruta.parent.mkdir(parents=True, exist_ok=True)

        # Guardar el resultado
        plt.imsave(ruta, pred_mask, cmap="gray")

        metrics = compute_metrics(pred_mask, gt_mask)

        all_ious.append(metrics["iou"])
        all_prec.append(metrics["precision"])
        all_rec.append(metrics["recall"])

        img_id = int(fname.split("_")[1].split(".")[0])
        if img_id >= 66:
            red_ious.append(metrics["iou"])
        else:
            green_ious.append(metrics["iou"])

    summary = {
        "iou_red": float(np.mean(red_ious)) if red_ious else 0.0,
        "iou_green": float(np.mean(green_ious)) if green_ious else 0.0,
        "iou_global": float(np.mean(all_ious)),
        "precision": float(np.mean(all_prec)),
        "recall": float(np.mean(all_rec)),
    }

    print(
        f"[{method_name}] "
        f"IoU Rojas: {summary['iou_red']:.4f} | "
        f"IoU Verdes: {summary['iou_green']:.4f} | "
        f"IoU Global: {summary['iou_global']:.4f} | "
        f"Prec: {summary['precision']:.4f} | "
        f"Rec: {summary['recall']:.4f}"
    )

    with open(f"results/{method_name}/metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary



if __name__ == "__main__":
    print("=== Resultados en TEST ===")

    segment_functions = {"Otsu": segment_otsu, 
                         "Watershed": segment_watershed}

    for method, fn in segment_functions.items():
        evaluate_method(TEST_DIR, fn, method)
        
    