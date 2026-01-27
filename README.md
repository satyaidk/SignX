# SignX

MediaPipe Hand pose Testing Model: https://storage.googleapis.com/tfjs-models/demos/hand-pose-detection/index.html?model=mediapipe_hands

# Folder & File Structure 

    sign_language_detector/
    │
    ├── dataset/                     # Raw videos (your input)
    │   ├── hello/
    │   │   ├── v1.mp4
    │   │   ├── v2.mp4
    │   │
    │   ├── hi/
    │   ├── how_are_you/
    │   ├── whats_up/
    │   └── you_good/
    │
    ├── processed_data/               # Extracted keypoints (numpy)
    │   ├── hello/
    │   ├── hi/
    │   ├── how_are_you/
    │   ├── whats_up/
    │   └── you_good/
    │
    ├── models/
    │   └── sign_model.h5
    │
├── src/
│   ├── config.py                 # Constants & labels
│   ├── extract_keypoints.py      # MediaPipe feature extraction
│   ├── process_videos.py         # Video → sequences
│   ├── train_model.py            # Model training
│   └── live_detection.py         # Webcam inference
│
├── requirements.txt
└── README.md

# Analogy:

You can think of this data preparation like a chef preparing ingredients for a large banquet. First, the chef catalogs all available ingredients (creating the DataFrame). Then, they set aside a small portion of the best ingredients to taste-test the final dish (the test set). Before cooking, they wash and chop everything into uniform sizes (normalization and resizing) and even toss them around to ensure they are well-mixed (augmentation), ensuring the final meal is consistent no-matter which part of the pot you scoop from.
