import cv2

tracker = cv2.legacy.TrackerCSRT_create()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        break

frame = cv2.flip(frame, 1)

bbox = cv2.selectROI("Select Object", frame, False)
tracker.init(frame, bbox)
cv2.destroyWindow("Select Object")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    success, box = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Tracking", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)
    else:
        cv2.putText(frame, "Lost Target! Press R to reselect.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255), 2)

        cv2.imshow("Object Tracking", frame)
        key = cv2.waitKey(500) & 0xFF

        if key == ord('r'):
            bbox = cv2.selectROI("Select Object", frame, False)
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, bbox)
            cv2.destroyAllWindows()
        continue

    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
