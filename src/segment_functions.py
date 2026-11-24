import cv2
import numpy as np



#OTSU
def segment_otsu(img_gray: np.ndarray) -> np.ndarray:
    """Segmentación binaria mediante umbral automático de Otsu."""
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask



def segment_watershed(img_gray: np.ndarray, dist_ratio: float = 0.4) -> np.ndarray:
    """Segmentación binaria mediante el algoritmo Watershed"""
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)

    # Limpieza de ruido y definición de fondo seguro
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Semillas interiores mediante transformada de distancia
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_3ch, markers)

    # markers == 1 es fondo, markers == -1 son fronteras, > 1 son manzanas
    return np.where(markers > 1, 255, 0).astype(np.uint8)