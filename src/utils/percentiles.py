import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGES_PATH = "data/images"


def compute_and_plot_percentiles(directory: str, threshold_b: float = 188.0) -> dict[str, tuple[float, float]]:
    a_green, b_green = [], []
    a_red, b_red = [], []

    for img_name in sorted(os.listdir(directory)):
        stem = os.path.splitext(img_name)[0]
        img_id = int(stem.split("_")[1])

        img_path = os.path.join(directory, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a, b = cv2.split(img_lab.astype(np.float32))

        if img_id < 66:  # Green apples
            a_green.append(np.percentile(a, 98))
            b_green.append(np.percentile(b, 98))
        else:  # Red apples
            a_red.append(np.percentile(a, 98))
            b_red.append(np.percentile(b, 98))

    stats = {
        "a_green": (float(np.mean(a_green)), float(np.std(a_green))),
        "b_green": (float(np.mean(b_green)), float(np.std(b_green))),
        "a_red": (float(np.mean(a_red)), float(np.std(a_red))),
        "b_red": (float(np.mean(b_red)), float(np.std(b_red))),
    }

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Histogram of b* (98th percentile) with threshold
    bins = np.linspace(170, 210, 22)
    axes[0].hist(b_green, bins=bins, alpha=0.65, color="#4CAF50", edgecolor="black", label="Green apples (id < 66)")
    axes[0].hist(b_red, bins=bins, alpha=0.65, color="#E53935", edgecolor="black", label="Red apples (id >= 66)")
    axes[0].axvline(threshold_b, color="black", linestyle="--", linewidth=2, label=f"Threshold b* = {threshold_b}")
    axes[0].set_title("98th Percentile Distribution of Channel b*")
    axes[0].set_xlabel("b* intensity (98th percentile)")
    axes[0].set_ylabel("Number of images")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # 2. 2D Scatter plot (a* vs b*) to spot boundary cases
    axes[1].scatter(a_green, b_green, color="#4CAF50", alpha=0.8, edgecolors="k", label="Green apples")
    axes[1].scatter(a_red, b_red, color="#E53935", alpha=0.8, edgecolors="k", label="Red apples")
    axes[1].axhline(threshold_b, color="black", linestyle="--", linewidth=2, label=f"Decision boundary (b* = {threshold_b})")
    axes[1].set_title("2D Feature Space (a* vs b* at 98th Percentile)")
    axes[1].set_xlabel("a* intensity (green -> red)")
    axes[1].set_ylabel("b* intensity (blue -> yellow)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("threshold_visualization.png", dpi=150)
    plt.show()

    return stats


if __name__ == "__main__":
    print(compute_and_plot_percentiles(IMAGES_PATH))