import cv2
import numpy as np

# HSV color ranges for red, yellow, and blue
color_ranges = {
    "red": [
        (np.array([0, 150, 150]), np.array([10, 255, 255])),
        (np.array([170, 150, 150]), np.array([180, 255, 255]))
    ],
    "yellow": [
        (np.array([20, 150, 150]), np.array([30, 255, 255]))
    ],
    "blue": [
        (np.array([100, 150, 150]), np.array([130, 255, 255]))
    ]
}

camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Pixel area range corresponding to 3cm to 10cm (adjust after calibration)
min_area = 60**2
max_area = 200**2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dominant_box = None
    dominant_color = None
    max_box_area = 0

    for color, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            temp_mask = cv2.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)

        # Noise removal
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area and area > max_box_area:
                max_box_area = area
                dominant_box = cv2.boundingRect(cnt)
                dominant_color = color

    # Draw bounding box if a box is detected
    if dominant_box is not None:
        x, y, w, h = dominant_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_color.upper() + " BOX", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Dominant Color Box Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
