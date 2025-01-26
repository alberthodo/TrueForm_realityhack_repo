import cv2
import mediapipe as mp
import numpy as np
import time
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import numpy as np


#Initialize OpenBCI
params = BrainFlowInputParams()
params.serial_port = "COM4"
board = BoardShim(BoardIds.CYTON_BOARD, params)

# Define Cyton Board ID
board_id = BoardIds.CYTON_BOARD.value
# Prepare and start the session

# initialize calibration and time variables
time_thres = 100
max_val = -100000000000
vals_mean = 0
num_samples = 5000
samples = 0

board = BoardShim(BoardIds.CYTON_BOARD, params)
board.prepare_session()

board.start_stream(45000)

flag = False
threshold = 2000  # Set the EMG threshold for Channel 1

# Get the index of Channel 1
eeg_channels = BoardShim.get_eeg_channels(board_id)
channel_1_index = eeg_channels[0]

# end calibration

    # start game
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

            data = board.get_current_board_data(32)  # Retrieve the last 32 data points (adjust as needed)

            # Extract Channel 1 data
            channel_1_data = data[channel_1_index]

            # EMG Preprocessing: Calculate absolute value
            abs_channel_1 = np.abs(channel_1_data)

            # Optional: Calculate RMS for a window
            rms_channel_1 = np.sqrt(np.mean(abs_channel_1 ** 2))

            # Check if the RMS value exceeds the threshold
            if rms_channel_1 > threshold:
                flag = True
            else:
                flag = False
            print(rms_channel_1)
            # Print the flag and RMS value
            print(f"Flag: {'ON' if flag else 'OFF'}, RMS Value: {rms_channel_1:.2f}")

            if shoulder and elbow:
                #Check EEG for Color
                cv2.line(mask, shoulder, elbow, (0, 255, 0), thickness=10)  # Blue line
                if flag == True:
                    cv2.line(mask, shoulder, elbow, (0, 0, 255), thickness=10)  # Blue line

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
