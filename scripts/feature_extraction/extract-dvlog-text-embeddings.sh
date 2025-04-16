#!/bin/bash

# Please set these variables
VIDEO_DIR=./data/D-vlog/videos/
WAV_DIR=./data/D-vlog/wavs/
NO_CHUNKED_DIR=./data/D-vlog/no-chunked/
DATA_DIR=./data/D-vlog/data/

# Extract WAV files from videos (if not already done)
python3 ./scripts/feature_extraction/dvlog/extract_wavs.py \
    --csv-path ./data/D-vlog/video_ids.csv \
    --column-video-id video_id \
    --video-dir $VIDEO_DIR \
    --dest-dir $WAV_DIR

# Extract text and text embeddings
python3 ./scripts/feature_extraction/dvlog/extract_text_embeddings.py \
    --audio-dir $WAV_DIR \
    --text-output-dir $NO_CHUNKED_DIR/text_transcripts/ \
    --embeddings-output-dir $NO_CHUNKED_DIR/text_embeddings/ \
    --batch-size 1 \
    --cuda-device cuda:0

# Process text embeddings to get no-text indices
python3 ./scripts/feature_extraction/dvlog/process_text_embeddings.py \
    --splits-dir ./data/D-vlog/splits/ \
    --data-dir $NO_CHUNKED_DIR \
    --text-embeddings-dir text_embeddings \
    --dest-dir no_text_idxs

# Split into chunks
python3 ./scripts/feature_extraction/dvlog/split_into_chunks.py \
    --source-dir $NO_CHUNKED_DIR \
    --dest-dir $DATA_DIR \
    --modality-id text_embeddings \
    --no-idxs-id no_text_idxs \
    --frame-rate 25 