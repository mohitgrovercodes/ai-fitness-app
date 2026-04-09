import cv2
import mediapipe as mp

class PostureModel:

    @staticmethod
    def process(file):
        # Convert to OpenCV image
        image = cv2.imdecode(file.file.read(), cv2.IMREAD_COLOR)

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        results = pose.process(image)

        if results.pose_landmarks:
            return {"posture": "Good"}
        return {"posture": "Not detected"}