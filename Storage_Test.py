import cv2
import numpy as np
import os
import time
import csv

# -------------------------------
# 0. CSV Storage Setup
# -------------------------------
CSV_FILE = "detections.csv"

# Create file if it doesn’t exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Logo", "Color"])

# Storage variables
detections = []            # current cycle detections
last_detection_time = time.time()
reset_wait = 5             # seconds of inactivity to reset system

# -------------------------------
# 1. Setup reference logos
# -------------------------------
logo_folder = "logos"
logo_names = ["amazon", "myntra", "nike", "zepto"]

ref_images = {}
for name in logo_names:
    img_path = os.path.join(logo_folder, f"{name}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARNING] Could not load {img_path}")
        continue
    ref_images[name] = img

# Initialize SIFT + ORB
sift = cv2.SIFT_create()
orb = cv2.ORB_create(2000)

ref_descriptors = {}
for name, img in ref_images.items():
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(des) < 5:
        kp, des = orb.detectAndCompute(img, None)
    if des is None:
        print(f"[WARNING] No descriptors found for {name}")
        continue
    ref_descriptors[name] = (kp, des, img.shape[::-1])

# FLANN matcher setup
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params_sift = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
index_params_orb = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)

flann_sift = cv2.FlannBasedMatcher(index_params_sift, search_params)
flann_orb = cv2.FlannBasedMatcher(index_params_orb, search_params)

def detect_logo(frame_gray):
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    matcher = flann_sift
    if des_frame is None or len(des_frame) < 5:
        kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)
        matcher = flann_orb

    if des_frame is None:
        return None, None, 0

    best_match = None
    best_good_matches = 0
    best_box = None

    for name, (kp_ref, des_ref, shape) in ref_descriptors.items():
        try:
            matches = matcher.knnMatch(des_ref, des_frame, k=2)
        except:
            continue

        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > 8:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                w, h = shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                if len(good_matches) > best_good_matches:
                    best_match = name
                    best_good_matches = len(good_matches)
                    best_box = dst.astype(int)

    return best_match, best_box, best_good_matches

# HSV color ranges
color_ranges = {
    "red": [
        (np.array([0, 150, 150]), np.array([10, 255, 255])),
        (np.array([170, 150, 150]), np.array([180, 255, 255]))
    ],
    "yellow": [
        (np.array([20, 150, 150]), np.array([30, 255, 255]))
    ],
    "blue": [
        (np.array([90, 100, 50]), np.array([130, 255, 255]))
    ]
}

min_area = 60**2
max_area = 200**2

cap = cv2.VideoCapture(0)  # Use your webcam index
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Logo detection ---
    prediction, box, score = detect_logo(frame_gray)
    if prediction and box is not None and score > 15:
        cv2.polylines(frame, [box], True, (0, 255, 0), 3)
        cv2.putText(frame, f"{prediction} ({score})",
                    (box[0][0][0], box[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # --- Dominant color box detection ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dominant_box = None
    dominant_color = None
    max_box_area = 0

    for color, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            temp_mask = cv2.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area and area > max_box_area:
                max_box_area = area
                dominant_box = cv2.boundingRect(cnt)
                dominant_color = color

    if dominant_box is not None:
        x, y, w, h = dominant_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, dominant_color.upper() + " BOX", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # -------------------------------
    # Storage mechanism
    # -------------------------------
    if prediction and dominant_color:
        detections.append((prediction, dominant_color))
        last_detection_time = time.time()

    # Reset cycle if idle
    if time.time() - last_detection_time > reset_wait and detections:
        print("\n[STORED RESULTS]")
        for logo, color in detections:
            print(f"{logo}:{color}")

            # Save to CSV
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), logo, color])

        detections.clear()

    cv2.imshow("Logo + Color Box Detection", frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):   # ESC or 'q'
        break


cap.release()
cv2.destroyAllWindows()
