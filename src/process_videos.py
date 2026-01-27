import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import ACTIONS, SEQUENCE_LENGTH, DATASET_PATH, PROCESSED_PATH
from extract_keypoints import extract_keypoints

base_options_pose = python.BaseOptions(model_asset_path='models/pose_landmarker.task')
options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

base_options_hand = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand)
hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

for action in ACTIONS:
    os.makedirs(os.path.join(PROCESSED_PATH, action), exist_ok=True)

    for video in os.listdir(f"{DATASET_PATH}/{action}"):
        cap = cv2.VideoCapture(f"{DATASET_PATH}/{action}/{video}")
        sequence = []

        while cap.isOpened() and len(sequence) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

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

        cap.release()

        if len(sequence) == SEQUENCE_LENGTH:
            np.save(
                f"{PROCESSED_PATH}/{action}/{video.split('.')[0]}.npy",
                sequence
            )

print("✅ Video processing complete")
