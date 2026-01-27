import numpy as np

def extract_keypoints(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    pose = np.array([[p.x, p.y, p.z]
                     for p in pose_landmarks]).flatten() \
                     if pose_landmarks else np.zeros(33*3)

    lh = np.array([[l.x, l.y, l.z]
                   for l in left_hand_landmarks]).flatten() \
                   if left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[r.x, r.y, r.z]
                   for r in right_hand_landmarks]).flatten() \
                   if right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])
