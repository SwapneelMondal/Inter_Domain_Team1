import cv2

for i in range(5):  # check first 5 camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        cap.release()
