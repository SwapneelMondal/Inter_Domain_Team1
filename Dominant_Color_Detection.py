import cv2
import numpy as np
from sklearn.cluster import KMeans

# Basic color dictionary (BGR)
COLOR_NAMES = {
    "Red": (0,0,255),
    "Green": (0,255,0),
    "Blue": (255,0,0),
    "Yellow": (0,255,255),
    "Cyan": (255,255,0),
    "Magenta": (255,0,255),
    "White": (255,255,255),
    "Black": (0,0,0),
    "Orange": (0,165,255),
    "Pink": (203,192,255),
    "Purple": (128,0,128)
}

def closest_color_name(bgr_color):
    """Find the nearest color name from the dictionary"""
    min_dist = float('inf')
    color_name = None
    for name, c in COLOR_NAMES.items():
        dist = np.linalg.norm(np.array(c) - np.array(bgr_color))
        if dist < min_dist:
            min_dist = dist
            color_name = name
    return color_name

def dominant_color(image, k=3):
    """Return dominant BGR color using KMeans"""
    img = cv2.resize(image, (150,150))
    img = img.reshape((-1,3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return tuple(map(int, dominant))

# -------------------------------
# Webcam demo
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    color = dominant_color(frame, k=3)
    color_name = closest_color_name(color)

    # Draw a rectangle with dominant color
    swatch = np.zeros((100,100,3), dtype=np.uint8)
    swatch[:] = color

    cv2.putText(frame, f"Dominant: {color_name} {color}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Dominant Color", swatch)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
