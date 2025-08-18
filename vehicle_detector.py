import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

minimum_width_rect = 20
minimum_height_rect = 40
max_width_rect = 70
count_line_position = 300

algo2 = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

detect = []
offset = 6
counter_lane1 = 0
counter_lane2 = 0
counter_lane3 = 0

# Define lane boundaries 
lane1 = (0, 232)
lane2 = (233, 399)
lane3 = (400, 640)

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (15, 15),0)
    img_sub = algo2.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilateada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilateada = cv2.morphologyEx(dilateada, cv2.MORPH_CLOSE, kernel)

    counterShape, _ = cv2.findContours(dilateada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw lane lines
    cv2.line(frame1, (lane1[0], count_line_position), (lane1[1], count_line_position), (255, 0, 0), 2)
    cv2.line(frame1, (lane2[0], count_line_position), (lane2[1], count_line_position), (0, 255, 0), 2)
    cv2.line(frame1, (lane3[0], count_line_position), (lane3[1], count_line_position), (0, 0, 255), 2)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        if not (w >= minimum_width_rect and h >= minimum_height_rect and w <= max_width_rect):
            continue

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

    for (cx, cy) in detect:
        if (cy < count_line_position + offset) and (cy > count_line_position - offset):
            if lane1[0] <= cx <= lane1[1]:
                counter_lane1 += 1
            elif lane2[0] <= cx <= lane2[1]:
                counter_lane2 += 1
            elif lane3[0] <= cx <= lane3[1]:
                counter_lane3 += 1
            detect.remove((cx, cy))

    # Display counters
    cv2.putText(frame1, f"Lane 1: {counter_lane1}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame1, f"Lane 2: {counter_lane2}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame1, f"Lane 3: {counter_lane3}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video original', frame1)
    # cv2.imshow('Detector', dilateada)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
