import cv2
import numpy as np

# Tighter HSV ranges for dominant colors
color_ranges = {
    "red": [
        (np.array([0, 150, 150]), np.array([10, 255, 255])),   # lower red
        (np.array([170, 150, 150]), np.array([180, 255, 255])) # upper red
    ],
    "yellow": [
        (np.array([20, 150, 150]), np.array([30, 255, 255]))   # bright yellow
    ],
    "blue": [
        (np.array([100, 150, 150]), np.array([130, 255, 255])) # deep blue
    ]
}

camera_index = 0
cap = cv2.VideoCapture(camera_index)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dominant_box = None
    dominant_color = None
    max_area = 0

    for color, ranges in color_ranges.items():
        mask = None

        for i, (lower, upper) in enumerate(ranges):
            temp_mask = cv2.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1500 and area > max_area:  # Pick only the largest among all colors
                max_area = area
                dominant_box = cv2.boundingRect(cnt)
                dominant_color = color

    if dominant_box is not None:
        x, y, w, h = dominant_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_color.upper() + " BOX", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Dominant Color Box Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
