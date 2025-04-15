#!/bin/bash

# Install required packages from requirements file
pip install -r landmark_requirements.txt

# Install git-lfs for handling large files
conda install -c conda-forge git-lfs
git lfs install

# Create feature extractors directory
mkdir -p ./feature_extractors/
cd ./feature_extractors/

# Clone and install face detection
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..

# Clone and install face alignment
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..

# Download MediaPipe models
wget -O ./body_pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
wget -O ./hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

cd ..
