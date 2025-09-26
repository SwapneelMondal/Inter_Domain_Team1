import cv2
import numpy as np
import os
import serial
import time

# -------------------------------
# 0. Setup Arduino Serial
# -------------------------------
ARDUINO_PORT = "COM6"
BAUD_RATE = 9600

try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[INFO] Connected to Arduino on {ARDUINO_PORT}.")
except Exception as e:
    arduino = None
    print(f"[WARNING] Could not connect to Arduino on {ARDUINO_PORT}: {e}")
    print("[WARNING] Servo & Motor control will be disabled.")

servo_active = False
servo_trigger_time = 0
servo_hold_time = 8

motor_active = False
motor_stop_time = 0

# -------------------------------
# 1. User Input (Target Company)
# -------------------------------
target_company = input("Enter company name (amazon/myntra/nike/zepto): ").strip().lower()

# -------------------------------
# 2. Setup reference logos
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

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params_sift = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
index_params_orb = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6, key_size=12, multi_probe_level=1)
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
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)
                if len(good_matches) > best_good_matches:
                    best_match = name
                    best_good_matches = len(good_matches)
                    best_box = dst.astype(int)
    return best_match, best_box, best_good_matches

# -------------------------------
# 3. Color detection
# -------------------------------
color_ranges = {
    "red":[(np.array([0,150,150]),np.array([10,255,255])),
           (np.array([170,150,150]),np.array([180,255,255]))],
    "yellow":[(np.array([20,150,150]),np.array([30,255,255]))],
    "blue":[(np.array([90,100,50]),np.array([130,255,255]))]
}

min_area = 60**2
max_area = 200**2

# -------------------------------
# 4. Detection & counters
# -------------------------------
required_detection_time = 2 # seconds

company_timers = {name: None for name in logo_names}
company_last_color = {name: None for name in logo_names}
company_total_counters = {name: 0 for name in logo_names}
company_color_counters = {name: {} for name in logo_names}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_time = time.time()

    # --- Logo detection ---
    prediction, box, score = detect_logo(frame_gray)
    detected_company = prediction.lower() if prediction else None

    if detected_company and box is not None and score>15:
        cv2.polylines(frame,[box],True,(0,255,0),3)
        cv2.putText(frame,f"{prediction} ({score})",(box[0][0][0],box[0][0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # initialize timer if first time
        if company_timers[detected_company] is None:
            company_timers[detected_company] = current_time
        elif current_time - company_timers[detected_company] >= required_detection_time:
            # --- LOGGING COUNTERS ---
            company_total_counters[detected_company] += 1
            total_cnt = company_total_counters[detected_company]

            color = company_last_color[detected_company]
            if color:
                if color not in company_color_counters[detected_company]:
                    company_color_counters[detected_company][color] = 0
                company_color_counters[detected_company][color] += 1
                color_cnt = company_color_counters[detected_company][color]
                print(f"{color_cnt} {detected_company.upper()} {color.upper()} box detected")

            print(f"{total_cnt} {detected_company.upper()} box detected")

            # --- Servo & Motor for ALL detected logos ---
            if not servo_active:
                print(f"[ACTION] {detected_company.upper()} detected → Servo OPEN")
                if arduino:
                    arduino.write(b"OPEN\n")
                servo_active = True
                servo_trigger_time = current_time

            if not motor_active:
                print(f"[ACTION] {detected_company.upper()} detected → Motor ON")
                if arduino:
                    arduino.write(b"MOTOR_ON\n")
                motor_active = True
                motor_stop_time = current_time + 10  # Motor runs 10s

            company_timers[detected_company] = current_time  # restart timer for next increment
    else:
        # reset timers if logo disappears
        for name in logo_names:
            company_timers[name] = None

    # Reset servo after hold time
    if servo_active and (current_time - servo_trigger_time > servo_hold_time):
        print("[ACTION] Servo CLOSE")
        if arduino:
            arduino.write(b"CLOSE\n")
        servo_active = False

    # Reset motor after 10 seconds
    if motor_active and current_time >= motor_stop_time:
        print("[ACTION] Motor OFF")
        if arduino:
            arduino.write(b"MOTOR_OFF\n")
        motor_active = False

    # --- Color detection ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dominant_box = None
    dominant_color = None
    max_box_area = 0

    for color, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            temp_mask = cv2.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)

        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area<=area<=max_area and area>max_box_area:
                max_box_area = area
                dominant_box = cv2.boundingRect(cnt)
                dominant_color = color

    if dominant_box:
        x,y,w,h = dominant_box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,dominant_color.upper()+" BOX",(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
        if detected_company:
            company_last_color[detected_company] = dominant_color

    cv2.imshow("Logo + Color Detection + Servo + Motor",frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
