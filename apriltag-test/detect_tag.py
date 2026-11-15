import cv2
import numpy as np
from apriltag import apriltag

class AprilTagDetector:
    def __init__(self, tag_size=0.165, webcam_idx=0):
        self.TAG_SIZE = tag_size # (m)
        self.detector = apriltag("tag36h11")
        self.webcam = cv2.VideoCapture(webcam_idx)

        if not self.webcam.isOpened():
            raise RuntimeError("could not open webcam")

        ret, frame = self.webcam.read()
        if not ret:
            raise RuntimeError("could not capture frame")

        height, width = frame.shape[:2]
        self.fx = float(width)
        self.fy = float(width)
        self.cx = width / 2 
        self.cy = height / 2

    def capture(self):
        ret, frame = self.webcam.read()
        if not ret:
            raise RuntimeError("could not capture frame")
        return frame

    def detect(self, frame):
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(grayscale_frame)
        results = []

        for detection in detections:
            pose = self.detector.estimate_tag_pose(
                detection, 
                self.TAG_SIZE,
                self.fx, 
                self.fy, 
                self.cx, 
                self.cy
            )

            R = pose["R"]
            t = pose["t"].reshape(3, 1)
            camera_pos = R.T @ t

            results.append({ 
                "id": detection["id"],
                "corners": detection["lb-rb-rt-lt"],
                "camera_pos": camera_pos.flatten(),
                "distance": np.linalg.norm(camera_pos)
            })
            print(results[-1])
            
        return results

    def cleanup(self):
        self.webcam.release()
        cv2.destroyAllWindows()
    