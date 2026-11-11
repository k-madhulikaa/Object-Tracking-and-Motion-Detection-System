import cv2

video = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)

MIN_AREA = 2000

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    blurred = cv2.GaussianBlur(frame, (7,7), 0)

    fgmask = fgbg.apply(blurred)
    _, thresh = cv2.threshold(fgmask, 240, 255, cv2.THRESH_BINARY)

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            motion_found = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0,255,0), 2)

    if motion_found:
        cv2.putText(frame, "Motion Detected!",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)
    else:
        cv2.putText(frame, "No Motion",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
