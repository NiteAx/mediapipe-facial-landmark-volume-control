import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import GUID

# Define the GUID for IAudioEndpointVolume
IID_IAudioEndpointVolume = GUID('{5CDF2C82-841E-4546-9722-0CF74078229A}')

# Get the default audio device (speakers)
devices = AudioUtilities.GetSpeakers()

# Activate the IAudioEndpointVolume interface for the default audio device
interface = devices.Activate(IID_IAudioEndpointVolume, CLSCTX_ALL, None)

# Cast the interface pointer to a POINTER(IAudioEndpointVolume) object
volume = cast(interface, POINTER(IAudioEndpointVolume))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Linearly decrease volume
target_vol = 0.90
speed = 6.0

# Initialize the holistic model
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB color space
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the holistic model to get the pose landmarks
        results = holistic.process(image)
        pose_landmarks = results.pose_landmarks

        # If pose landmarks are detected
        if pose_landmarks is not None:
            # Calculate the yaw, pitch, and roll angles
            rvec, _ = cv2.Rodrigues(np.array([pose_landmarks.landmark[0].visibility, pose_landmarks.landmark[0].x, pose_landmarks.landmark[0].y]))
            angles, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = angles[:, 0]

            # Draw the yaw, pitch, and roll angles on the frame
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if pitch < 0.40:
                # Set the master volume level to 10%
                if target_vol > 0.1:
                    target_vol = target_vol - 0.01*speed
                    print("Lowering volume.")
                else:
                    print("Volume set to 10%.")
            else:
                # Set the master volume level to 90%
                if target_vol < 0.9:
                    target_vol = target_vol + 0.01*speed
                    print("Increasing volume.")
                else:
                    print("Volume set to 90%.")

            volume.SetMasterVolumeLevelScalar(target_vol, None)
        else:
            print("Muting volume.")
            volume.SetMasterVolumeLevelScalar(0.0, None)
        # Display the frame in a window
        #cv2.imshow("Holistic Model", frame)

        # Exit the loop if the "q" key is pressed
        if cv2.waitKey(1) == ord("q"):
            break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()