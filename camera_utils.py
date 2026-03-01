
import cv2
import mediapipe as mp
import math
from typing import Dict, Any, Tuple
STALE_SECONDS = 2.0  

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def _make_unknown_data() -> Dict[str, Any]:
    return {
        "posture": "unknown",
        "confidence": 0.0,
        "source": "camera",
        "raw_text": "no_data",
        "received_time": 0.0,
    }

def _calculate_angle(a, b, c) -> float:
    """Return angle in degrees at point b given three normalized landmark-like points with .x/.y."""
    ang = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) -
        math.atan2(a.y - b.y, a.x - b.x)
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang

def _classify_from_landmarks(landmarks) -> Tuple[str, float]:
    """Given a mediapipe landmark list, compute mid-shoulder / mid-hip and decide posture.
       Returns (payload_string, angle)."""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    mid_shoulder = [(left_shoulder.x + right_shoulder.x) / 2.0,
                    (left_shoulder.y + right_shoulder.y) / 2.0]
    mid_hip = [(left_hip.x + right_hip.x) / 2.0,
               (left_hip.y + right_hip.y) / 2.0]

    class P:
        def __init__(self, x, y): self.x = x; self.y = y

    shoulder = P(*mid_shoulder)
    hip = P(*mid_hip)
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    angle = _calculate_angle(nose, shoulder, hip)
    # same threshold logic as you used before:
    if angle > 175:
        payload = "upright"
    else:
        payload = "slouched"

    return payload, angle



