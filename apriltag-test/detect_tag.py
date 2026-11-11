import cv2
import numpy as np
from apriltag import apriltag

def webcam_detect_tag():
    detector = apriltag("tag36h11")
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()

        if not ret:
            print("failed to capture frame")
            break

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(grayscale_frame)

        for detection in detections:
            corners = detection["lb-rb-rt-lt"].astype(int)

            cv2.polylines(
                frame,
                [corners],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            tag_id = detection["id"]
            cv2.putText(
                frame,
                f"ID: {tag_id}",
                tuple(corners[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        cv2.imshow("AprilTag Detection", frame)

        # 'q' to esc
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()
    return