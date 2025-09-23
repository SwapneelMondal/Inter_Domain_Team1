import cv2
import os

# -------------------------------
# 1. Setup reference logos
# -------------------------------
logo_folder = "logos"
logo_names = ["Amazon", "Blinkit", "Flipkart", "Reddit"]

# Load reference images with error handling
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

# -------------------------------
# 2. Function to match logo
# -------------------------------
def identify_logo(test_img_path):
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        print(f"[ERROR] Could not load test image: {test_img_path}")
        return None, 0

    kp_test, des_test = sift.detectAndCompute(test_img, None)
    if des_test is None:
        print("[ERROR] No descriptors found in test image.")
        return None, 0

    # FLANN matcher for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    best_match = None
    best_good_matches = 0

    for name, (kp_ref, des_ref) in ref_descriptors.items():
        if des_ref is None:
            continue

        matches = flann.knnMatch(des_ref, des_test, k=2)

        # Lowe’s ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > best_good_matches:
            best_good_matches = len(good_matches)
            best_match = name

    return best_match, best_good_matches

# -------------------------------
# 3. Example usage
# -------------------------------
if __name__ == "__main__":
    test_logo = "Amazon.png"  # Replace with your test logo
    prediction, score = identify_logo(test_logo)

    if prediction is None:
        print("[FAIL] Could not identify logo.")
    else:
        print(f"Predicted Logo: {prediction} (Good matches: {score})")

        # Display with annotation
        img = cv2.imread(test_logo)
        if img is None:
            print("[WARNING] Cannot display test image (failed to load).")
        else:
            cv2.putText(img, f"{prediction} ({score})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Prediction", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
