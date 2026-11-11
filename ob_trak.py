import cv2

# Create CSRT tracker (high accuracy)
tracker = cv2.legacy.TrackerCSRT_create()

# Start webcam
cap = cv2.VideoCapture(0)

# Read first valid frame
while True:
    ret, frame = cap.read()
    if ret:
        break

# Mirror the frame
frame = cv2.flip(frame, 1)

# Let user select object to track
bbox = cv2.selectROI("Select Object", frame, False)
tracker.init(frame, bbox)
cv2.destroyWindow("Select Object")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for natural tracking
    frame = cv2.flip(frame, 1)

    # Update tracker to get new bounding box
    success, box = tracker.update(frame)

    if success:
        # Draw tracked object
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Tracking", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        # Tracking failed
        cv2.putText(frame, "Lost Target! Press R to reselect.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Object Tracking", frame)
        key = cv0.waitKey(500) & 0xFF

        # Allow user to reselect object
        if key == ord('r'):
            bbox = cv2.selectROI("Select Object", frame, False)
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, bbox)
            cv2.destroyAllWindows()
        continue

    # Show tracking output
    cv2.imshow("Object Tracking", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
