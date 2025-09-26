import cv2
import numpy as np
import os
import time
import csv
import serial

# -------------------------------
# 0. Arduino Setup
# -------------------------------
ARDUINO_PORT = "COM6"
BAUD_RATE = 9600
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[INFO] Connected to Arduino on {ARDUINO_PORT}")
except Exception as e:
    arduino = None
    print(f"[WARNING] Could not connect to Arduino: {e}")
    print("[WARNING] Servo control will be disabled")

servo_active = False
servo_trigger_time = 0
servo_hold_time = 5  # seconds

# -------------------------------
# 1. CSV Storage Setup
# -------------------------------
CSV_FILE = "detections.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Logo", "Color", "LogoBox", "ColorBox"])

ready_to_store = True
stable_detection = None
stable_start_time = None
stable_time_required = 1.0  # seconds
reset_wait = 3
last_detection_time = time.time()
amazon_count = 0

# -------------------------------
# 2. Logo Loading
# -------------------------------
logo_names = ["Adidas", "Amazon", "Nike", "Zepto"]
logo_folder = "logos"
logo_folder_abs = os.path.abspath(logo_folder)
print(f"[INFO] Loading logos from: {logo_folder_abs}")

ref_images = {}
for name in logo_names:
    found = False
    for ext in [".png", ".jpg", ".jpeg"]:
        img_path = os.path.join(logo_folder_abs, f"{name}{ext}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            ref_images[name] = img
            print(f"[INFO] Loaded {img_path} ({img.shape[1]}x{img.shape[0]})")
            found = True
            break
    if not found:
        print(f"[ERROR] Could NOT load {name} logo.")

print(f"[INFO] Successfully loaded logos: {list(ref_images.keys())}")

# -------------------------------
# 3. Feature Extraction Setup
# -------------------------------
sift = cv2.SIFT_create()
orb = cv2.ORB_create(3000)
ref_descriptors = {}
for name, img in ref_images.items():
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(des) < 5:
        kp, des = orb.detectAndCompute(img, None)
    if des is None:
        print(f"[WARNING] No descriptors for {name}")
        continue
    ref_descriptors[name] = (kp, des, img.shape[::-1])

# FLANN matcher
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
flann_sift = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))
flann_orb = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1), dict(checks=50))

# -------------------------------
# 4. Logo Detection Function (Single Logo, Safe Homography)
# -------------------------------
def detect_single_logo(frame_gray):
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    matcher = flann_sift
    if des_frame is None or len(des_frame) < 5:
        kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)
        matcher = flann_orb
    if des_frame is None:
        return None, None, 0

    best_match = None
    best_box = None
    best_score = 0

    for name, (kp_ref, des_ref, shape) in ref_descriptors.items():
        try:
            matches = matcher.knnMatch(des_ref, des_frame, k=2)
        except:
            continue
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # Only compute homography if at least 4 good matches exist
        if len(good_matches) >= 4:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                w, h = shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)
                # Choose the logo with the most good matches
                if len(good_matches) > best_score:
                    best_match = name
                    best_box = dst.astype(int)
                    best_score = len(good_matches)

    return best_match, best_box, best_score

# -------------------------------
# 5. Color Detection Setup
# -------------------------------
color_ranges = {
    "red":[(np.array([0,150,150]), np.array([10,255,255])),
           (np.array([170,150,150]), np.array([180,255,255]))],
    "yellow":[(np.array([20,150,150]), np.array([30,255,255]))],
    "blue":[(np.array([90,100,50]), np.array([130,255,255]))]
}
min_area = 60**2
max_area = 200**2

# -------------------------------
# 6. Main Loop
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Logo Detection (single logo) ---
    prediction, box, score = detect_single_logo(frame_gray)
    current_detections = []
    if prediction and box is not None:
        cv2.polylines(frame, [box], True, (0,255,0), 3)
        cv2.putText(frame, f"{prediction} ({score})", (box[0][0][0], box[0][0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        current_detections.append((prediction, box))

    # --- Color Detection ---
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
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area and area > max_box_area:
                max_box_area = area
                dominant_box = cv2.boundingRect(cnt)
                dominant_color = color
    if dominant_box is not None:
        x,y,w,h = dominant_box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, dominant_color.upper()+" BOX", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # --- Store single detection in CSV ---
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for name, box in current_detections:
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                name,
                dominant_color if dominant_color else "",
                box.flatten().tolist() if box is not None else "",
                dominant_box if dominant_box else ""
            ])

    # --- Stability + Amazon Servo ---
    if prediction == "Amazon" and dominant_color:
        current_detection = ("Amazon", dominant_color)
        if stable_detection == current_detection:
            elapsed = time.time() - stable_start_time
            if elapsed >= stable_time_required and ready_to_store:
                print(f"[INFO] Amazon:{dominant_color} stored")
                ready_to_store = False
                if arduino:
                    arduino.write(b"OPEN\n")
                    servo_active = True
                    servo_trigger_time = time.time()
                    amazon_count += 1
                    lcd_msg = f"Amazon:{amazon_count}\n"
                    arduino.write(lcd_msg.encode())
        else:
            stable_detection = current_detection
            stable_start_time = time.time()
    else:
        stable_detection = None
        stable_start_time = None

    # --- Reset mechanism ---
    if not current_detections:
        if time.time() - last_detection_time > reset_wait:
            ready_to_store = True
            last_detection_time = time.time()
    else:
        last_detection_time = time.time()

    # --- Servo auto-close ---
    if servo_active and time.time() - servo_trigger_time > servo_hold_time:
        print("[ACTION] Servo CLOSE")
        if arduino:
            arduino.write(b"CLOSE\n")
        servo_active = False

    cv2.imshow("Logo + Color Box Detection", frame)
    key = cv2.waitKey(1)
    if key==27 or key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
