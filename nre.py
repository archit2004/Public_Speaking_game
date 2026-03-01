import cv2
import mediapipe as mp
import math
import socket

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort=("127.0.0.1", 12345)
def calculate_angle(a, b, c):
    """Returns angle (in degrees) at point b given 3 landmarks (a-b-c)."""
    ang = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) -
        math.atan2(a.y - b.y, a.x - b.x)
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang

def classify_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Use average points for mid-shoulder and mid-hip
    mid_shoulder = [(left_shoulder.x + right_shoulder.x)/2,
                    (left_shoulder.y + right_shoulder.y)/2]
    mid_hip = [(left_hip.x + right_hip.x)/2,
               (left_hip.y + right_hip.y)/2]

    # Convert to "fake landmark objects" to reuse angle function
    class P: 
        def __init__(self,x,y): self.x=x; self.y=y
    shoulder = P(*mid_shoulder)
    hip = P(*mid_hip)
    # Use nose as reference for upper point
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Angle: Nose–Shoulder–Hip (forward lean detection)
    angle = calculate_angle(nose, shoulder, hip)

    if angle > 170:  # almost straight
        sock.sendto(b'Upright', serverAddressPort)
        return "Upright"
    else:
        sock.sendto(b'Slouched', serverAddressPort)
        return "Slouched"

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            posture = classify_posture(landmarks)

            mp_drawing.draw_landmarks(frame,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, posture, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Posture Classification', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
