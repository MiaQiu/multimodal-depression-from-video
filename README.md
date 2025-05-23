<h1 align="center"><span style="font-weight:normal">Reading Between the 🎞️ Frames:<br />Multi-Modal Depression Detection in Videos from Non-Verbal Cues</h1>
<h2 align="center"> Accepted at the 2024 edition of European Conference on Information Retrieval (ECIR).</h2>

<div align="center">
  
[David Gimeno-Gómez](https://scholar.google.es/citations?user=DVRSla8AAAAJ&hl=en), [Ana-Maria Bucur](https://scholar.google.com/citations?user=TQuQ5IAAAAAJ&hl=en), [Adrian Cosma](https://scholar.google.com/citations?user=cdYk_RUAAAAJ&hl=en), [Carlos-D. Martínez-Hinarejos](https://scholar.google.es/citations?user=M_EmUoIAAAAJ&hl=en), [Paolo Rosso](https://scholar.google.es/citations?user=HFKXPH8AAAAJ&hl=en)
</div>



<div align="center">
  
[📘 Introduction](#intro) |
[🛠️ Data Preparation](#preparation) |
[💪 Training and Evaluation](#training) |
[📖 Citation](#citation) |
[📝 License](#license)
</div>

<div align="center"> <img src="images/arch.png"  width="720"> </div>

## <a name="tldr"> </a> TL;DR 
<div align="center">
  
  [📜 Arxiv Link](https://arxiv.org/abs/2401.02746)
  [🛝 Google Slides](https://docs.google.com/presentation/d/1QskN661UrygIYqaqa1yGkuLz4O4zwbnal_gWO5lOJ0s/edit?usp=sharing)
</div>

We extract high-level non-verbal cues using **pretrained models**, process them using a **modality-specific encoder**, condition the resulting embeddings with **positional and modality embeddings**, and process the sequence with a **transformer encoder** to perform the final classification.

## <a name="intro"></a> 📘 Introduction
*Depression, a prominent contributor to global disability, affects a substantial portion of the population. Efforts to detect depression from social media texts have been prevalent, yet only a few works explored depression detection from user-generated video content. In this work, we address this research gap by proposing a simple and flexible multi-modal temporal model capable of discerning non-verbal depression cues from diverse modalities in noisy, real-world videos. We show that, for in-the-wild videos, using additional high-level non-verbal cues is crucial to achieving good performance, and we extracted and processed audio speech embeddings, face emotion embeddings, face, body and hand landmarks, and gaze and blinking information. Through extensive experiments, we show that our model achieves state-of-the-art results on three key benchmark datasets for depression detection from video by a substantial margin.*

## <a name="preparation"></a> 🛠️ Data Preparation

### Downloading the datasets

- For D-Vlog, the features extracted by the authors are publicly available [here](https://sites.google.com/view/jeewoo-yoon/dataset). Original vlog videos are available upon request. Please contact the original paper authors.

- For DAIC-WOZ and E-DAIC, the features are only available upon request [here](https://dcapswoz.ict.usc.edu/).

### Extracting non-verbal modalities

<details>
<summary> Click here for detailed tutorial </summary>

#### D-Vlog

- To extract the audio embeddings:

```
conda create -y -n pase+ python=3.7
conda activate pase+
bash ./scripts/conda_envs/prepare_pase+_env.sh
bash ./scripts/features
scripts/feature_extraction/extract-dvlog-pase+-feats.sh
conda deactivate pase+
```

- To extract face, body, and hand landmarks:

```
conda create -y -n landmarks python=3.8
conda activate landmarks
bash scripts/conda_envs/prepare_landmarks_env.sh
scripts/feature_extraction/extract-dvlog-landmarks.sh
conda deactivate landmarks
```

- To extract face EmoNet embeddings:

```
conda create -y -n emonet python=3.8
conda activate emonet
bash ./scripts/conda_envs/prepare_emonet_env.sh
bash ./scripts/feature_extraction/extract-dvlog-emonet-feats.sh
conda deactivate emonet
```

The EmoNet feature extraction:
- Uses the EmoNet model from "Estimation of continuous valence and arousal levels from faces in naturalistic conditions" (Nature Machine Intelligence, 2021)
- Extracts 256-dimensional embeddings from the last layer before classification
- Processes faces in batches for GPU acceleration
- Supports both the 8-class model (neutral, happy, sad, surprise, fear, disgust, anger, contempt) and the 5-class model
- Maintains temporal alignment with other modalities at 25 fps
- Features are saved in compressed numpy format (.npz)

This feature extraction pipeline:
1. Uses previously extracted face crops
2. Passes each face through the EmoNet model
3. Captures the 256-dimensional embedding vector from the last layer before classification
4. Saves features in the appropriate format for the multi-modal depression model

- To extract Wav2Vec2 audio embeddings:

```
conda create -y -n wav2vec2 python=3.8
conda activate wav2vec2
pip install torch torchaudio transformers pandas pyannote.audio numpy tqdm
bash ./scripts/feature_extraction/extract-dvlog-wav2vec2-feats.sh
conda deactivate wav2vec2
```

The Wav2Vec2 feature extraction:
- Uses the facebook/wav2vec2-base model
- Processes audio files in batches (default: 4 files per batch)
- Extracts 768-dimensional features at 25fps
- Supports GPU acceleration (set via --cuda-device)
- Automatically handles audio resampling to 16kHz
- Saves features in compressed numpy format (.npz)

Requirements:
- Hugging Face authentication token (required for voice activity detection)
- Directory structure:
  - `./data/D-vlog/videos/` - Original video files
  - `./data/D-vlog/wavs/` - Extracted WAV files
  - `./data/D-vlog/no-chunked/` - Unchunked feature files
  - `./data/D-vlog/data/` - Final processed features
- Video IDs CSV file at `./data/D-vlog/video_ids.csv`

The extraction pipeline:
1. Extracts WAV files from videos
2. Detects voice activity using pyannote.audio
3. Processes voice activity to identify non-voice segments
4. Extracts Wav2Vec2 features from audio
5. Splits features into chunks for training

- To extract text and text embeddings:

```
conda create -y -n text python=3.8
conda activate text
pip install torch torchaudio transformers numpy tqdm ffmpeg-python
bash ./scripts/feature_extraction/extract-dvlog-text-embeddings.sh
conda deactivate text
```

The text and text embeddings extraction:
- Uses OpenAI's Whisper model (whisper-medium) for speech-to-text transcription
- Uses BERT model fine-tuned on sentiment (nlptown/bert-base-multilingual-uncased-sentiment) for text embeddings
- Processes audio files in 30-second chunks with 5-second overlap
- Extracts two types of embeddings:
  - CLS token embeddings (768-dimensional) - good for classification tasks
  - Mean pooled embeddings (768-dimensional) - good for semantic representation
- Saves both transcripts and embeddings in separate directories

Requirements:
- Python packages:
  - torch and torchaudio - for audio processing and deep learning
  - transformers - for Whisper and BERT models
  - numpy - for array operations
  - tqdm - for progress bars
  - ffmpeg-python - for audio file handling
- Directory structure:
  - `./data/D-vlog/videos/` - Original video files
  - `./data/D-vlog/wavs/` - Extracted WAV files
  - `./data/D-vlog/no-chunked/` - Unchunked feature files
    - `text_transcripts/` - Transcribed text files
    - `text_embeddings/` - Text embeddings
  - `./data/D-vlog/data/` - Final processed features
- Video IDs CSV file at `./data/D-vlog/video_ids.csv`

The extraction pipeline:
1. Extracts WAV files from videos (if not already done)
2. Transcribes audio to text using Whisper with English language setting
3. Extracts text embeddings using BERT
4. Processes embeddings to identify non-text segments
5. Splits features into chunks for training

Output format:
- Transcripts: Plain text files (.txt)
- Embeddings: NumPy files (.npy)
  - CLS embeddings: 768-dimensional vectors
  - Mean embeddings: 768-dimensional vectors

- To extract gaze tracking:

```
conda create -y -n mpiigaze python=3.8
conda activate mpiigaze
bash ./scripts/conda_envs/prepare_mpiigaze_env.sh
bash ./scripts/feature_extraction/extract-dvlog-gaze-feats.sh
conda deactivate mpiigaze
```

- To extract blinking features:

```
conda create -y -n instblink python=3.7
conda activate instblink
bash ./scripts/conda_envs/prepare_instblink_env.sh
bash ./scripts/feature_extraction/extract-dvlog-blinking-feats.sh
conda deactivate instblink
```
#### DAIC-WOZ

- To pre-process the DAIC-WOZ features:

```
conda activate landmarks
bash ./scripts/feature_extraction/extract-daicwoz-features.sh
conda deactivate
```

#### E-DAIC
- To pre-process the DAIC-WOZ features:

```
conda activate landmarks
bash ./scripts/feature_extraction/extract-edaic-features.sh
conda deactivate
```

</details>

### Implementation Detail

Once all the data has been pre-processed, you should indicate the absule path to the directory where it is stored
in the 'configs/env_config.yaml' file for each one of the corresponding datasets.

In addition, you can continue working in the 'landmarks' environment, since it has everything we 
need for training and evaluating our model:

```
conda activate landmarks
```

### Modality distributions for D-Vlog

Once extracted, the modalities for D-Vlog should be similar to the following plots (counts / missing frames):


<div align="center"> <img src="images/counts.png"  width="256"> <img src="images/presence-fraction.png"  width="256"> </div>

## <a name="training"></a> 💪 Training and Evaluation
To train and evaluate the models and the results reported in the paper, you can run the following commands:

```
cd experiments/
bash run-exps.sh
```

## <a name="citation"></a> 📖 Citation
If you found our work useful, please cite our paper:

[Reading Between the Frames: Multi-Modal Non-Verbal Depression Detection in Videos](https://arxiv.org/abs/2401.02746)

```
@InProceedings{gimeno2024videodepression,
  author="Gimeno-G{\'o}mez, David and Bucur, Ana-Maria and Cosma, Adrian and Mart{\'i}nez-Hinarejos, Carlos-David and Rosso, Paolo",
  editor="Goharian, Nazli and Tonellotto, Nicola and He, Yulan and Lipani, Aldo and McDonald, Graham and Macdonald, Craig and Ounis, Iadh",
  title="Reading Between the Frames: Multi-modal Depression Detection in Videos from Non-verbal Cues",
  booktitle="Advances in Information Retrieval",
  year="2024",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="191--209",
  isbn="978-3-031-56027-9"
}
```

This repository is based on the [Acumen ✨ Template ✨](https://github.com/cosmaadrian/acumen-template).

## <a name="license"></a> 📝 License

This work is protected by [CC BY-NC-ND 4.0 License (Non-Commercial & No Derivatives)](LICENSE)
