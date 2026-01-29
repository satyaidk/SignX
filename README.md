# SignX

MediaPipe Hand pose Testing Model: https://storage.googleapis.com/tfjs-models/demos/hand-pose-detection/index.html?model=mediapipe_hands

# Folder & File Structure 
```
sign_model_phase_1/
├── .gitignore
├── README.md
├── requirements.txt
├── dataset/
│   ├── hello/hello.mp4
│   ├── hi/hi.mp4
│   ├── how_are_you/how_are_you.mp4
│   ├── whats_up/whats_up.mp4
│   └── you_good/you_good.mp4
├── processed_data/
│   ├── hello/hello.npy
│   ├── hi/hi.npy
│   ├── how_are_you/how_are_you.npy
│   ├── whats_up/whats_up.npy
│   └── you_good/you_good.npy
├── models/
│   ├── hand_landmarker.task
│   ├── holistic_landmarker.task
│   ├── pose_landmarker.task
│   └── sign_model.h5
└── src/
    ├── config.py
    ├── extract_keypoints.py
    ├── process_videos.py
    ├── train_model.py
    └── live_detection.py
```
# Analogy:

You can think of this data preparation like a chef preparing ingredients for a large banquet. First, the chef catalogs all available ingredients (creating the DataFrame). Then, they set aside a small portion of the best ingredients to taste-test the final dish (the test set). Before cooking, they wash and chop everything into uniform sizes (normalization and resizing) and even toss them around to ensure they are well-mixed (augmentation), ensuring the final meal is consistent no-matter which part of the pot you scoop from.
