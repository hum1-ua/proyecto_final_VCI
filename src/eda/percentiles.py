import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

TRAIN_IMAGES_DIR = "data/train/images"
PARAMS_PATH = "data/preprocess_params.json"
PLOT_PATH = "eda/train_b99_calibration.png"


def extract_train_percentiles(
    images_dir: str,
    percentile: int
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Computes 99th percentile of a* and b* channels for every image in images_dir."""
    filenames, a_p99_list, b_p99_list, is_red_list = [], [], [], []

    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith(".png"):
            continue

        img_bgr = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_COLOR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        img_smooth = cv2.bilateralFilter(img_lab, d=9, sigmaColor=25, sigmaSpace=50)

        filenames.append(fname)
        a_p99_list.append(np.percentile(img_smooth[:, :, 1], percentile))
        b_p99_list.append(np.percentile(img_smooth[:, :, 2], percentile))

        img_id = int(fname.split("_")[1].split(".")[0])
        is_red_list.append(img_id >= 66)

    return (
        filenames,
        np.array(a_p99_list),
        np.array(b_p99_list),
        np.array(is_red_list, dtype=bool),
    )


def build_calibration_figure(
    filenames: list[str],
    a_p99: np.ndarray,
    b_p99: np.ndarray,
    is_red: np.ndarray,
    percentile: int, 
    thresh: float | None = None,
    interactive: bool = False,
    
) -> plt.Figure:
    """Builds histogram and 2D feature space plots with optional hover tooltips."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names_arr = np.array(filenames)

    # Left: 99th percentile distribution of channel b*
    bins = np.arange(np.floor(b_p99.min()) - 1, np.ceil(b_p99.max()) + 2, 1.5)
    axes[0].hist(
        b_p99[~is_red], bins=bins, color="#4daf4a", alpha=0.75, edgecolor="black", label="Green apples (id < 66)"
    )
    axes[0].hist(
        b_p99[is_red], bins=bins, color="#e41a1c", alpha=0.75, edgecolor="black", label="Red apples (id >= 66)"
    )
    if thresh is not None:
        axes[0].axvline(thresh, color="black", linestyle="--", linewidth=2, label=f"Threshold b* = {thresh:.2f}")
    axes[0].set_title(f"{percentile}th Percentile Distribution of Channel b* (Train Set)")
    axes[0].set_xlabel(f"b* intensity ({percentile}th percentile)")
    axes[0].set_ylabel("Number of images")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right: 2D Feature space (a* vs b* at 99th percentile)
    sc_green = axes[1].scatter(
        a_p99[~is_red], b_p99[~is_red], color="#4daf4a", edgecolor="black", s=45, alpha=0.85, label="Green apples"
    )
    sc_red = axes[1].scatter(
        a_p99[is_red], b_p99[is_red], color="#e41a1c", edgecolor="black", s=45, alpha=0.85, label="Red apples"
    )
    if thresh is not None:
        axes[1].axhline(
            thresh, color="black", linestyle="--", linewidth=2, label=f"Decision boundary (b* = {thresh:.2f})"
        )
    axes[1].set_title(f"2D Feature Space (a* vs b* at {percentile}th Percentile - Train Set)")
    axes[1].set_xlabel("a* intensity (green -> red)")
    axes[1].set_ylabel("b* intensity (blue -> yellow)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if interactive:
        green_names = names_arr[~is_red]
        red_names = names_arr[is_red]

        annot = axes[1].annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
            zorder=10,
        )
        annot.set_visible(False)

        def on_hover(event) -> None:
            if event.inaxes != axes[1]:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return

            matched_labels = []
            target_pos = None

            for sc, subset_names in ((sc_green, green_names), (sc_red, red_names)):
                contained, ind = sc.contains(event)
                if contained:
                    if target_pos is None:
                        target_pos = sc.get_offsets()[ind["ind"][0]]
                    for idx in ind["ind"]:
                        matched_labels.append(subset_names[idx])

            if matched_labels and target_pos is not None:
                annot.xy = target_pos
                x_mid = 0.5 * sum(axes[1].get_xlim())
                annot.set_position((-125, 12) if target_pos[0] > x_mid else (12, 12))

                header = f"(a*={target_pos[0]:.1f}, b*={target_pos[1]:.1f})"
                annot.set_text(header + "\n" + "\n".join(matched_labels))
                annot.set_visible(True)
                fig.canvas.draw_idle()
            elif annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    PERCENTILE = 99
    
    filenames, a_p99, b_p99, is_red = extract_train_percentiles(TRAIN_IMAGES_DIR, percentile=PERCENTILE)

    # 1. Show interactive distribution with hover labels so you can inspect outliers
    fig_raw = build_calibration_figure(filenames, a_p99, b_p99, is_red, thresh=None, interactive=True, percentile=PERCENTILE)
    plt.show()
    plt.close(fig_raw)

    # 2. Enter your chosen threshold in the terminal
    chosen_t = float(input("Introduce el umbral elegido para b* (percentil 99): "))
    train_acc = float(np.mean((b_p99 < chosen_t) == is_red))

    # 3. Save clean final plot with your boundary and store parameter in JSON
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)

    fig_final = build_calibration_figure(filenames, a_p99, b_p99, is_red, percentile = PERCENTILE, thresh=chosen_t, interactive=False)
    fig_final.savefig(PLOT_PATH, dpi=300)
    plt.close(fig_final)

    with open(PARAMS_PATH, "w") as f:
        json.dump({"b_p99_thresh": chosen_t, "train_accuracy": train_acc}, f, indent=2)

    print(f"Umbral guardado: {chosen_t:.2f} | Acierto resultante en Train: {train_acc * 100:.2f}%")
    print(f"Gráfica guardada en {PLOT_PATH}")