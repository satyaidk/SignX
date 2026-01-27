import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from config import ACTIONS, SEQUENCE_LENGTH, MODEL_PATH
from extract_keypoints import extract_keypoints

model = load_model(MODEL_PATH)

base_options_pose = python.BaseOptions(model_asset_path='models/pose_landmarker.task')
options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

base_options_hand = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand)
hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

sequence = []
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose_result = pose_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    left_hand = None
    right_hand = None
    for i, hand in enumerate(hand_result.hand_landmarks):
        handedness = hand_result.handedness[i]
        if isinstance(handedness, list):
            handedness = handedness[0]
        if handedness.category_name == 'Left':
            left_hand = hand
        elif handedness.category_name == 'Right':
            right_hand = hand

    keypoints = extract_keypoints(pose, left_hand, right_hand)
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:]

    if len(sequence) == SEQUENCE_LENGTH:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        action = ACTIONS[np.argmax(res)]

        cv2.putText(frame, action,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("Sign Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
