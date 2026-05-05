import cv2
import mediapipe as mp
import numpy as np
from app.utils.logger import logger

class PostureModel:
    # Initialize MediaPipe Pose as a class attribute (Singleton pattern)
    try:
        import mediapipe.python.solutions.pose as mp_pose
        _mp_pose = mp_pose
    except (ImportError, AttributeError):
        try:
            _mp_pose = mp.solutions.pose
        except AttributeError:
            logger.error("❌ MediaPipe Pose solution not found. Ensure mediapipe is installed correctly.")
            _mp_pose = None

    if _mp_pose:
        _pose = _mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    else:
        _pose = None

    @classmethod
    def process(cls, file):
        """Processes an image file to detect posture landmarks."""
        if cls._pose is None:
            return {"posture": "MediaPipe not initialized", "status": "error"}
            
        try:
            # 1. Read bytes and convert to numpy buffer
            file_bytes = file.file.read()
            if not file_bytes:
                return {"posture": "Empty file", "status": "error"}

            nparr = np.frombuffer(file_bytes, np.uint8)
            
            # 2. Decode image (OpenCV returns BGR)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return {"posture": "Invalid image", "status": "error"}

            # 3. Convert BGR to RGB (MediaPipe requirement)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 4. Process with MediaPipe
            results = cls._pose.process(image_rgb)

            if results.pose_landmarks:
                logger.info("✅ Posture detected successfully.")
                return {"posture": "Good", "status": "success"}
            
            logger.warning("⚠️ No pose landmarks detected.")
            return {"posture": "Not detected", "status": "success"}

        except Exception as e:
            logger.error(f"❌ Posture processing error: {str(e)}")
            return {"posture": "Error", "details": str(e), "status": "error"}