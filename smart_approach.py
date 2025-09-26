import cv2
import numpy as np
import os

# -------------------------------
# 1. Setup reference logos
# -------------------------------
logo_folder = "logos"
logo_names = ["Amazon", "Flipkart", "jio", "Nike", "Test"]

# Load reference images
ref_images = {}
for name in logo_names:
    img_path = os.path.join(logo_folder, f"{name}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARNING] Could not load {img_path}")
        continue
    ref_images[name] = img

# Initialize SIFT
sift = cv2.SIFT_create()

# Compute descriptors for reference logos
ref_descriptors = {}
for name, img in ref_images.items():
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(des) < 2:
        print(f"[WARNING] Not enough descriptors for {name}")
        continue
    ref_descriptors[name] = (kp, des, img.shape[::-1])  # (keypoints, descriptors, (width, height))

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# -------------------------------
# 2. Detect logo & return bounding box
# -------------------------------
def detect_logo(frame_gray):
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    if des_frame is None or len(des_frame) < 2:
        return None, None, 0

    best_match = None
    best_good_matches = 0
    best_box = None

    for name, (kp_ref, des_ref, shape) in ref_descriptors.items():
        if des_ref is None or len(des_ref) < 2:
            continue

        matches = flann.knnMatch(des_ref, des_frame, k=2)
        if len(matches) == 0:
            continue

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) > 10:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                w, h = shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)
                if len(good_matches) > best_good_matches:
                    best_match = name
                    best_good_matches = len(good_matches)
                    best_box = dst.astype(int)

    return best_match, best_box, best_good_matches

# -------------------------------
# 3. Webcam loop
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prediction, box, score = detect_logo(frame_gray)

    if prediction and box is not None and score > 40:
        # Draw bounding polygon
        cv2.polylines(frame, [box], True, (0,255,0), 3)

        # Put label above the first corner
        cv2.putText(frame, f"{prediction} ({score})", 
                    (box[0][0][0], box[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        # -----------------------------
        # Small square diagonally outside the detected logo
        # -----------------------------
        offset = 5  # pixels away from logo
        square_size = 5  # 5x5 region
        # bottom-right corner of the detected polygon
        x = box[2][0][0] + offset
        y = box[2][0][1] + offset

        # Ensure square is inside frame
        if x + square_size < frame.shape[1] and y + square_size < frame.shape[0]:
            square = frame[y:y+square_size, x:x+square_size]
            avg_color = square.mean(axis=(0,1))  # BGR average
            avg_color_int = tuple([int(c) for c in avg_color])
            
            # Compute average color in BGR
            square = frame[y:y+square_size, x:x+square_size]
            avg_color = square.mean(axis=(0,1))  # BGR average
            avg_color_int = tuple([int(c) for c in avg_color])

            # Convert BGR -> RGB
            avg_color_rgb = (avg_color_int[2], avg_color_int[1], avg_color_int[0])

            # Draw small square and display color on frame (still blue rectangle)
            cv2.rectangle(frame, (x, y), (x+square_size, y+square_size), (255,0,0), 2)
            cv2.putText(frame, str(avg_color_rgb), (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            # Print RGB to console
            print(f"Detected color (RGB): {avg_color_rgb}")

            # Draw small square and display color
            cv2.rectangle(frame, (x, y), (x+square_size, y+square_size), (255,0,0), 2)
            cv2.putText(frame, str(avg_color_int), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            
    cv2.imshow("Logo Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
