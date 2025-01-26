import cv2
import mediapipe as mp
import numpy as np
import brainflow as bf
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
def checkEEG():
    bf.BrainFlowMetrics()

# Initialize MediaPipe Drawing (for debugging or optional landmarks visualization)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Create a mask to apply the green filter
    mask = np.zeros_like(frame, dtype=np.uint8)

    if results.pose_landmarks:
        # Iterate through arm landmarks (e.g., shoulders, elbows, wrists)
        arm_landmarks = [
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]

        h, w, _ = frame.shape
        landmark_coords = {}

        for landmark in arm_landmarks:
            lm = results.pose_landmarks.landmark[landmark]
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_coords[landmark] = (x, y)

            # Draw circles or apply a filter around the landmarks (e.g., green circles)
            cv2.circle(mask, (x, y), 15, (255, 0, 0), thickness=-1)

        # Draw blue lines connecting shoulder to elbow and elbow to wrist

            shoulder = landmark_coords.get(getattr(mp_pose.PoseLandmark, f"RIGHT_SHOULDER"))
            elbow = landmark_coords.get(getattr(mp_pose.PoseLandmark, f"RIGHT_ELBOW"))
            wrist = landmark_coords.get(getattr(mp_pose.PoseLandmark, f"RIGHT_WRIST"))

            if shoulder and elbow:
                #Check EEG for Color
                cv2.line(mask, shoulder, elbow, (0, 255, 0), thickness=10)  # Blue line
            if elbow and wrist:
                #Check EEG for Color
                cv2.line(mask, elbow, wrist, (0, 255, 0), thickness=10)  # Blue line

    # Combine the mask with the original frame
    green_filtered_frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)

    # Display the output
    cv2.imshow("Arm Parts Highlighted", green_filtered_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
