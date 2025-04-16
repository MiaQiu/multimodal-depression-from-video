import os
import torch
import torchaudio
import argparse
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import List, Tuple

def load_audio(audio_path, target_sr=16000):
    """Load and resample audio file to target sample rate."""
    try:
        print(f"Loading audio from {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        print(f"Original sample rate: {sr}, shape: {waveform.shape}")
        
        if sr != target_sr:
            print(f"Resampling from {sr} to {target_sr}")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze().numpy()
        print(f"Final waveform shape: {waveform.shape}")
        return waveform
    except Exception as e:
        print(f"Error in load_audio: {str(e)}")
        raise

def process_audio(audio_path, feature_extractor, model, device):
    """Process audio file and extract Wav2Vec2 features."""
    try:
        # Load and preprocess audio
        waveform = load_audio(audio_path)
        
        # Process audio through Wav2Vec2
        print("Processing audio through Wav2Vec2")
        inputs = feature_extractor(waveform, 
                                 sampling_rate=16000, 
                                 return_tensors="pt",
                                 padding=True)
        print(f"Feature extractor output keys: {inputs.keys()}")
        
        # Move inputs to device
        input_values = inputs.input_values.to(device)
        print(f"Input values shape: {input_values.shape}")
        
        # Extract features
        print("Extracting features")
        with torch.no_grad():
            outputs = model(input_values)
        
        # Get the last hidden state
        features = outputs.last_hidden_state
        print(f"Features shape: {features.shape}")
        
        return features.cpu().numpy()
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        raise

def process_video(audioID):
    """Process a single audio file and save its features."""
    try:
        audio_path = os.path.join(args.audio_dir, audioID)
        dst_embedding_path = os.path.join(args.audio_embeddings_output_dir, audioID.replace('.wav', '.npz'))

        if os.path.exists(dst_embedding_path):
            print(f"Skipping sample {audioID} because it was already processed")
            return

        print(f"\nProcessing {audioID}")
        # Extract features
        features = process_audio(audio_path, feature_extractor, model, args.cuda_device)
        
        # Save features
        print(f"Saving features to {dst_embedding_path}")
        np.savez_compressed(dst_embedding_path, data=features)
        print(f"Successfully processed {audioID}")
    except Exception as e:
        print(f"Error processing {audioID}: {str(e)}")
        raise

def load_audio_batch(audio_paths: List[str], target_sr: int = 16000) -> Tuple[np.ndarray, List[str]]:
    """Load and resample a batch of audio files to target sample rate."""
    waveforms = []
    valid_paths = []
    
    for audio_path in audio_paths:
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            waveforms.append(waveform.squeeze().numpy())
            valid_paths.append(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
    
    return np.array(waveforms), valid_paths

def process_audio_batch(audio_paths: List[str], feature_extractor, model, device) -> Tuple[np.ndarray, List[str]]:
    """Process a batch of audio files and extract Wav2Vec2 features."""
    try:
        # Load and preprocess audio
        waveforms, valid_paths = load_audio_batch(audio_paths)
        if len(waveforms) == 0:
            return np.array([]), []
        
        # Process audio through Wav2Vec2
        print(f"Processing batch of {len(waveforms)} audio files")
        inputs = feature_extractor(waveforms, 
                                 sampling_rate=16000, 
                                 return_tensors="pt",
                                 padding=True)
        
        # Move inputs to device
        input_values = inputs.input_values.to(device)
        
        # Extract features
        with torch.no_grad():
            outputs = model(input_values)
        
        # Get the last hidden state
        features = outputs.last_hidden_state.cpu().numpy()
        
        return features, valid_paths
    except Exception as e:
        print(f"Error in process_audio_batch: {str(e)}")
        return np.array([]), []

def process_video_batch(audio_paths: List[str], feature_extractor, model, device, output_dir: str):
    """Process a batch of audio files and save their features."""
    try:
        # Extract features for the batch
        features, valid_paths = process_audio_batch(audio_paths, feature_extractor, model, device)
        
        if len(features) == 0:
            return
        
        # Save features for each file in the batch
        for i, audio_path in enumerate(valid_paths):
            audio_id = os.path.basename(audio_path)
            dst_embedding_path = os.path.join(output_dir, audio_id.replace('.wav', '.npz'))
            
            if os.path.exists(dst_embedding_path):
                print(f"Skipping sample {audio_id} because it was already processed")
                continue
            
            # Save features for this file
            np.savez_compressed(dst_embedding_path, data=features[i])
            print(f"Successfully processed {audio_id}")
    except Exception as e:
        print(f"Error processing batch: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 features from audio files")
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--audio-embeddings-output-dir", type=str, required=True, help="Output directory for audio embeddings")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base", help="Wav2Vec2 model name")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of audio files to process in each batch")
    parser.add_argument("--left-index", type=int, default=0)
    parser.add_argument("--right-index", type=int, default=None)
    args = parser.parse_args()

    print("Initializing model and feature extractor")
    # Initialize model and feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(args.model_name).to(args.cuda_device)
    model.eval()

    # Create output directory
    os.makedirs(args.audio_embeddings_output_dir, exist_ok=True)

    # Get list of audio files
    audio_files = sorted([f for f in os.listdir(args.audio_dir) if f.endswith('.wav')])
    if args.right_index is not None:
        audio_files = audio_files[args.left_index:args.right_index]
    else:
        audio_files = audio_files[args.left_index:]

    print(f"Found {len(audio_files)} audio files to process")
    
    # Process files in batches
    for i in tqdm(range(0, len(audio_files), args.batch_size)):
        batch_files = audio_files[i:i + args.batch_size]
        batch_paths = [os.path.join(args.audio_dir, f) for f in batch_files]
        process_video_batch(batch_paths, feature_extractor, model, args.cuda_device, args.audio_embeddings_output_dir) 