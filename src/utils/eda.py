import os
import cv2
import numpy as np

IMAGES_PATH = "data/images"
MASKS_PATH = "data/masks"  # Adjust to your ground truth folder

fg_lab, bg_lab = [], []
fg_hsv, bg_hsv = [], []

for name in sorted(os.listdir(IMAGES_PATH)):
    img = cv2.imread(os.path.join(IMAGES_PATH, name))
    mask = cv2.imread(os.path.join(MASKS_PATH, name), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    fg = mask > 127
    bg = ~fg

    fg_lab.append(lab[fg].mean(axis=0))
    bg_lab.append(lab[bg].mean(axis=0))
    fg_hsv.append(hsv[fg].mean(axis=0))
    bg_hsv.append(hsv[bg].mean(axis=0))

print("Mean LAB (Apples):", np.mean(fg_lab, axis=0))
print("Mean LAB (Background):", np.mean(bg_lab, axis=0))
print("Mean HSV (Apples):", np.mean(fg_hsv, axis=0))
print("Mean HSV (Background):", np.mean(bg_hsv, axis=0))