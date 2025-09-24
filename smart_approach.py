import cv2
import numpy as np
import os

# -------------------------------
# 1. Setup reference logos
# -------------------------------
logo_folder = "logos"
logo_names = ["Amazon", "Blinkit", "Flipkart", "jio", "Nike"]

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
    if des is None:
        print(f"[WARNING] No descriptors found for {name}")
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
    if des_frame is None:
        return None, None, 0

    best_match = None
    best_good_matches = 0
    best_box = None

    for name, (kp_ref, des_ref, shape) in ref_descriptors.items():
        matches = flann.knnMatch(des_ref, des_frame, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) > 10:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                w, h = shape  # width, height
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

    if prediction and box is not None and score > 20:  # lowered threshold
        # Draw bounding polygon
        cv2.polylines(frame, [box], True, (0,255,0), 3)

        # Put label
        cv2.putText(frame, f"{prediction} ({score})", 
                    (box[0][0][0], box[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        # -------------------------------
        # Ratio-based offset for color detection
        # -------------------------------
        w = int(np.linalg.norm(box[0][0] - box[1][0]))  # width of logo
        h = int(np.linalg.norm(box[1][0] - box[2][0]))  # height of logo

        offset_x = int(0.2 * w)  # 20% of logo width
        offset_y = int(0.2 * h)  # 20% of logo height

        # Take bottom-right corner and move diagonally outward
        sample_x = int(box[2][0][0] + offset_x)
        sample_y = int(box[2][0][1] + offset_y)

        if 0 <= sample_x < frame.shape[1] and 0 <= sample_y < frame.shape[0]:
            color = frame[sample_y, sample_x].tolist()  # BGR
            rgb_color = (color[2], color[1], color[0])  # Convert BGR → RGB

            # Convert to HSV for filtering
            hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
            h_val, s_val, v_val = hsv_color

            # Filter out low saturation / brightness → Unknown
            if s_val < 50 or v_val < 50 or v_val > 240:
                closest = "Unknown"
                text_color = (128, 128, 128)  # gray text
            else:
                ref_colors = {
                    "Red": (255, 0, 0),
                    "Yellow": (255, 255, 0),
                    "Green": (0, 255, 0)
                }

                def closest_color(rgb):
                    distances = {name: np.linalg.norm(np.array(rgb) - np.array(val)) 
                                 for name, val in ref_colors.items()}
                    return min(distances, key=distances.get)

                closest = closest_color(rgb_color)
                text_color = (255, 0, 0)  # blue text for clarity

            # Draw square + label
            cv2.rectangle(frame, (sample_x-5, sample_y-5), (sample_x+5, sample_y+5), (255,0,0), 2)
            cv2.putText(frame, f"{closest} {rgb_color}", (sample_x+10, sample_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Print in console
            print(f"Detected RGB: {rgb_color} → Closest: {closest}")

    cv2.imshow("Logo Detection with Color Sampling", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
