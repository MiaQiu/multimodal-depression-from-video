import os
import sys
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import AutoTokenizer, AutoModel

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def transcribe_audio(audio_path, device):
    """Transcribe audio to text using Whisper."""
    model_id = "openai/whisper-medium"
    
    try:
        # Initialize model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline with return_timestamps=True for long-form audio
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            return_timestamps=True,
            chunk_length_s=30,  # Process in 30-second chunks
            stride_length_s=5   # 5-second overlap between chunks
        )
        
        # Transcribe audio with English language
        result = transcriber(audio_path, generate_kwargs={"language": "en"})
        return result["text"]
    except Exception as e:
        logging.error(f"Error transcribing {audio_path}: {str(e)}")
        return None

def get_mental_bert_embeddings(text, device):
    """Extract text embeddings using BERT."""
    # Using a public BERT model fine-tuned on emotion/sentiment
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        
        # Tokenize text
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract embeddings
        cls_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        mean_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return {
            "cls_embedding": cls_embedding,
            "mean_embedding": mean_embedding
        }
    except Exception as e:
        logging.error(f"Error extracting embeddings: {str(e)}")
        return None

def process_audio_file(audio_path, text_output_path, embeddings_output_path, device):
    """Process a single audio file to extract text and embeddings."""
    # First transcribe audio to text
    transcript = transcribe_audio(audio_path, device)
    if transcript is None:
        return False
    
    # Save transcript
    with open(text_output_path, 'w') as f:
        f.write(transcript)
    
    # Then extract Mental-BERT embeddings
    embeddings = get_mental_bert_embeddings(transcript, device)
    if embeddings is None:
        return False
    
    # Save embeddings
    np.save(os.path.join(embeddings_output_path, "cls.npy"), embeddings['cls_embedding'])
    np.save(os.path.join(embeddings_output_path, "mean.npy"), embeddings['mean_embedding'])
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Script to extract text and text embeddings from WAV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing WAV files"
    )
    parser.add_argument(
        "--text-output-dir",
        type=str,
        required=True,
        help="Directory to save transcribed text"
    )
    parser.add_argument(
        "--embeddings-output-dir",
        type=str,
        required=True,
        help="Directory to save text embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="cuda:0",
        help="CUDA device to use"
    )

    args = parser.parse_args()
    setup_logging()

    # Verify input directory exists
    if not os.path.exists(args.audio_dir):
        logging.error(f"Audio directory not found: {args.audio_dir}")
        sys.exit(1)

    # Create output directories
    os.makedirs(args.text_output_dir, exist_ok=True)
    os.makedirs(args.embeddings_output_dir, exist_ok=True)

    # Set device
    device = args.cuda_device if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Get list of WAV files
    wav_files = [f for f in os.listdir(args.audio_dir) if f.endswith('.wav')]
    logging.info(f"Found {len(wav_files)} WAV files to process")

    # Process each WAV file
    successful = 0
    failed = 0

    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        base_name = os.path.splitext(wav_file)[0]
        audio_path = os.path.join(args.audio_dir, wav_file)
        text_path = os.path.join(args.text_output_dir, f"{base_name}.txt")
        embeddings_path = os.path.join(args.embeddings_output_dir, base_name)
        
        # Create embeddings directory for this file
        os.makedirs(embeddings_path, exist_ok=True)
        
        # Process the file
        if process_audio_file(audio_path, text_path, embeddings_path, device):
            successful += 1
        else:
            failed += 1

    # Print summary
    logging.info(f"""
Processing complete:
- Successful: {successful}
- Failed: {failed}
- Total processed: {len(wav_files)}
""")

if __name__ == "__main__":
    main() 