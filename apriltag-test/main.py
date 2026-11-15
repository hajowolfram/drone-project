import cv2
from detect_tag import AprilTagDetector

def main():
    detector = AprilTagDetector(tag_size=0.165, webcam_idx=0)

    while True:
        frame = detector.capture()
        results = detector.detect(frame)

        for result in results:
            corners = result["corners"].astype(int)
            x, y, z = result["camera_pos"]
            
            cv2.polylines(
                frame,
                [corners],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )
            cv2.putText(
                frame,
                f"ID: {result['id']} | "
                f"({x:.2f}, {y:.2f}, {z:.2f}) m",
                tuple(corners[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow("AprilTag Localization", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                detector.cleanup()
                break

if __name__ == "__main__":
    main()