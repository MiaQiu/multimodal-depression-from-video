#!/bin/bash

# Please set these variables
USE_AUTH_TOKEN=your_huggingface_token_here

VIDEO_DIR=./data/D-vlog/videos/
WAV_DIR=./data/D-vlog/wavs/
NO_CHUNKED_DIR=./data/D-vlog/no-chunked/
DATA_DIR=./data/D-vlog/data/

# Extract WAV files from videos
python3 ./scripts/feature_extraction/dvlog/extract_wavs.py \
    --csv-path ./data/D-vlog/video_ids.csv \
    --column-video-id video_id \
    --video-dir $VIDEO_DIR \
    --dest-dir $WAV_DIR

# Extract Wav2Vec2 features
python3 ./scripts/feature_extraction/dvlog/extract_wav2vec2_features.py \
    --audio-dir $WAV_DIR \
    --audio-embeddings-output-dir $NO_CHUNKED_DIR/wav2vec2_embeddings/ \
    --batch-size 4 \
    --cuda-device cuda:0

# Extract voice activity
python3 ./scripts/feature_extraction/dvlog/extract_audio_activity.py \
    --pretrained-model pyannote/voice-activity-detection \
    --use-auth-token $USE_AUTH_TOKEN \
    --audio-dir $WAV_DIR \
    --dest-dir $NO_CHUNKED_DIR/audio_activity/

# Process voice activity to get no-voice indices
python3 ./scripts/feature_extraction/dvlog/process_voice_activity.py \
    --splits-dir ./data/D-vlog/splits/ \
    --data-dir $NO_CHUNKED_DIR \
    --audio-embeddings-dir wav2vec2_embeddings \
    --audio-activity-dir $NO_CHUNKED_DIR/audio_activity/ \
    --dest-dir no_voice_idxs



# Split into chunks
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py \
    --source-dir $NO_CHUNKED_DIR \
    --dest-dir $DATA_DIR \
    --modality-id wav2vec2_embeddings \
    --no-idxs-id no_voice_idxs \
    --frame-rate 25 