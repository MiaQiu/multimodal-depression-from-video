#!/bin/bash

# Set the variables
EMONET_CHCK_PATH=./feature_extractors/emonet/pretrained/emonet_8.pth

VIDEO_DIR=./data/D-vlog/videos/
WAV_DIR=./data/D-vlog/wavs/
NO_CHUNKED_DIR=./data/D-vlog/no-chunked/
DATA_DIR=./data/D-vlog/data/

# Create necessary directories
mkdir -p $NO_CHUNKED_DIR/face_emonet_embeddings/
mkdir -p $DATA_DIR

# FACE EMBEDDINGS - using the 256-dimensional embeddings from the last layer before classification
# This captures the 256-dimensional embedding vector as described in the paper
python3 ./scripts/feature_extraction/dvlog/extract_face_emonet_features.py \
    --checkpoint $EMONET_CHCK_PATH \
    --faces-dir $NO_CHUNKED_DIR/faces/ \
    --face-embeddings-output-dir $NO_CHUNKED_DIR/face_emonet_embeddings/ \
    --n-expression 8 \
    --cuda-device cuda \
    --batch-size 64 \
    --n-jobs 4 \
    --process-every-n-frames 1  # Set to higher value (e.g., 2 or 3) to process fewer frames for faster results

# Split the embeddings into chunks for training
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py \
    --source-dir $NO_CHUNKED_DIR \
    --dest-dir $DATA_DIR \
    --modality-id face_emonet_embeddings \
    --no-idxs-id no_face_idxs \
    --frame-rate 25

echo "EmoNet feature extraction completed!"
