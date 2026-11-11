import cv2

# Start webcam (camera index 0)
video = cv2.VideoCapture(0)

# Create a background subtractor objects
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)

MIN_AREA = 2000  # Minimum area of movement to consider as real motion

while True:
    #read a frame from the webcam
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view

    blurred = cv2.GaussianBlur(frame, (7,7), 0)  # Reduce noise

    fgmask = fgbg.apply(blurred)  # Apply background subtraction to get moving regions
    _, thresh = cv2.threshold(fgmask, 240, 255, cv2.THRESH_BINARY)

    # remove small noise using erosion and dilation
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # Find moving object contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAN_APPROX_SIMPLE
    )

    motion_found = False

    # Check each contour's size
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            motion_found = True
            x, y, w, h = cv2.boundingRect(cnt)  # Draw motion box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # Display motion status
    if motion_found:
        cv2.putText(frame, "Motion Detected!",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv2.putText(frame, "No Motion",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

video.release()
cv2.destroyAllWindows()
