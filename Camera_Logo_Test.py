import cv2
import os

# -------------------------------
# 1. Setup reference logos
# -------------------------------
logo_folder = "logos"
logo_names = ["Amazon", "Blinkit", "Flipkart", "Myntra"]

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
    ref_descriptors[name] = (kp, des)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# -------------------------------
# 2. Function to identify logo
# -------------------------------
def identify_logo(frame_gray):
    kp_test, des_test = sift.detectAndCompute(frame_gray, None)
    if des_test is None:
        return None, 0

    best_match = None
    best_good_matches = 0

    for name, (kp_ref, des_ref) in ref_descriptors.items():
        if des_ref is None:
            continue
        matches = flann.knnMatch(des_ref, des_test, k=2)
        # Lowe’s ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good_matches) > best_good_matches:
            best_good_matches = len(good_matches)
            best_match = name

    return best_match, best_good_matches

# -------------------------------
# 3. Start webcam
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prediction, score = identify_logo(frame_gray)
    
    if prediction:
        text = f"{prediction} ({score})"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        text2 = "It is true"
    if score > 40:
        cv2.putText(
            frame,
            text2,                        # your text
            (500, 300),                     # position (x,y)
            cv2.FONT_HERSHEY_SIMPLEX,     # fontFace
            1,                            # fontScale
            (0, 255, 0),                  # color (green)
            2,                            # thickness
            cv2.LINE_AA                   # line type
        )
    cv2.imshow("Real-Time Logo Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
