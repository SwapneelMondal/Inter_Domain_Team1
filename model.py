import cv2
import numpy as np
import os

# -------------------------------
# COLOR DETECTION FROM FIRST CODE (EXPANDED RANGES)
# -------------------------------
color_ranges = {
    "red": [
        (np.array([0, 50, 50]), np.array([15, 255, 255])),
        (np.array([165, 50, 50]), np.array([180, 255, 255]))
    ],
    "yellow": [
        (np.array([15, 50, 50]), np.array([35, 255, 255]))
    ],
    "blue": [
        (np.array([90, 50, 50]), np.array([140, 255, 255]))
    ]
}

# Area constraints for colored boxes - more flexible
min_area = 40**2
max_area = 350**2

# -------------------------------
# EXACT 4 COMPANIES ONLY
# -------------------------------
logo_names = ["amazon", "myntra", "nike", "zepto"]
logo_folder = "logos"

ref_images = {}
for name in logo_names:
    img_path = os.path.join(logo_folder, f"{name}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        ref_images[name] = img
        print(f"[INFO] Loaded {name}")
    else:
        print(f"[ERROR] Missing {img_path}")

if len(ref_images) != 4:
    print("[ERROR] Not all 4 logo files found!")
    exit()

# -------------------------------
# Feature detection setup
# -------------------------------
sift = cv2.SIFT_create(nfeatures=800)
orb = cv2.ORB_create(nfeatures=1500)

ref_descriptors = {}
for name, img in ref_images.items():
    # Try SIFT first
    kp, des = sift.detectAndCompute(img, None)
    detector = "sift"
    
    # Fall back to ORB if needed
    if des is None or len(des) < 8:
        kp, des = orb.detectAndCompute(img, None)
        detector = "orb"
    
    if des is not None and len(des) >= 8:
        ref_descriptors[name] = (kp, des, detector)
        print(f"[INFO] {name}: {len(des)} features ({detector})")

# FLANN matchers
flann_sift = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=5), 
    dict(checks=50)
)
flann_orb = cv2.FlannBasedMatcher(
    dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), 
    dict(checks=50)
)

# -------------------------------
# LOGO DETECTION FROM SECOND CODE (STRICTER/BETTER)
# -------------------------------
def detect_logo(roi_gray):
    best_match = None
    best_score = 0
    best_box = None
    
    # Get features from ROI
    kp_sift, des_sift = sift.detectAndCompute(roi_gray, None)
    kp_orb, des_orb = orb.detectAndCompute(roi_gray, None)
    
    for name, (kp_ref, des_ref, detector) in ref_descriptors.items():
        try:
            # Match with same detector type
            if detector == "sift" and des_sift is not None and len(des_sift) >= 8:
                matches = flann_sift.knnMatch(des_ref, des_sift, k=2)
                kp_frame = kp_sift
            elif detector == "orb" and des_orb is not None and len(des_orb) >= 8:
                if des_orb.dtype != np.uint8:
                    des_orb = des_orb.astype(np.uint8)
                if des_ref.dtype != np.uint8:
                    des_ref = des_ref.astype(np.uint8)
                matches = flann_orb.knnMatch(des_ref, des_orb, k=2)
                kp_frame = kp_orb
            else:
                continue
            
            # Filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Need at least 10 good matches
            if len(good_matches) >= 10:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None and mask is not None:
                    inliers = np.sum(mask)
                    if inliers > best_score and inliers >= 8:
                        best_score = inliers
                        best_match = name
                        
                        # Get bounding box
                        h, w = ref_images[name].shape
                        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                        best_box = cv2.perspectiveTransform(pts, M)
                        
        except Exception as e:
            continue
    
    return best_match, best_box, best_score

# -------------------------------
# Main loop with FIRST CODE's color detection
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Find the largest colored box - USING FIRST CODE'S BETTER COLOR DETECTION
    best_box = None
    best_color = None
    largest_area = 0
    
    for color_name, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            temp_mask = cv2.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)
        
        # Clean up the mask - improved from first code
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area and area > largest_area:
                largest_area = area
                best_box = cv2.boundingRect(contour)
                best_color = color_name
    
    # Process the best colored box found
    if best_box is not None:
        x, y, w, h = best_box
        
        # Extract ROI and detect logo
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        logo_name, logo_box, score = detect_logo(roi_gray)
        
        # Draw the colored box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if logo_name and logo_box is not None and score >= 8:
            # Adjust logo box coordinates to full frame
            logo_box[:, 0, 0] += x
            logo_box[:, 0, 1] += y
            cv2.polylines(frame, [logo_box.astype(int)], True, (0, 255, 255), 3)
            
            label = f"{best_color.upper()} BOX + {logo_name.upper()}"
        else:
            label = f"{best_color.upper()} BOX"
        
        # Draw label
        cv2.putText(frame, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Debug: Show all detected colors from first code
    debug_y = 30
    for color_name, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            temp_mask = cv2.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)
        
        detected_pixels = cv2.countNonZero(mask)
        color_debug = (0, 255, 0) if detected_pixels > 1000 else (0, 0, 255)
        cv2.putText(frame, f"{color_name}: {detected_pixels}", (10, debug_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_debug, 1)
        debug_y += 20
    
    cv2.imshow("Dominant Color + Logo Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()