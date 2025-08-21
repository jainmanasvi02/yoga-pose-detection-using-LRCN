# yoga-pose-detection-using-LRCN

This project implements a real-time yoga pose detection system using Python and computer vision techniques. It leverages a Long-term Recurrent Convolutional Network (LRCN) — a hybrid deep learning model combining Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal sequence learning.

#Features

-Accurate Pose Recognition: Detects and classifies 10 yoga poses with high precision.
-Real-Time Analysis: Processes video streams for instant feedback.
-Spatial-Temporal Learning: Uses CNN + LSTM for both posture recognition and motion dynamics.
-Posture Alignment Support: Provides immediate insights to help improve yoga practice.
-Practical Applications: Personal fitness tracking, virtual yoga coaching, and rehabilitation support.

#System Architecture
Input (Live Webcam / Video)
        ↓
Preprocessing (OpenCV + MediaPipe)
        ↓
Feature Extraction (CNN)
        ↓
Sequence Learning (LSTM)
        ↓
Prediction (Yoga Pose Classification)
        ↓
Real-Time Accuracy Percentage


#This code only has the lrcn model which demonstrated the neural netwrok layers used to make the project.
