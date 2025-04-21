#!/bin/bash

echo "Please follow the indications in https://github.com/face-analysis/emonet"

# Install required packages for EmoNet based on the official requirements
echo "Installing required packages for EmoNet..."

# Core packages required according to the README:
# - PyTorch (tested on version 1.2.0)
# - OpenCV (tested on version 4.1.0)
# - skimage (tested on version 0.15.0)
# - face-alignment 

# Install core dependencies with specific versions
pip install torch==1.2.0 torchvision==0.4.0
pip install opencv-python==4.1.0.25
pip install scikit-image==0.15.0
pip install numpy pandas matplotlib tqdm joblib

# Create feature extractors directory if it doesn't exist
mkdir -p ./feature_extractors/
cd ./feature_extractors/

# Clone and install face-alignment repository (required by EmoNet)
if [ ! -d "face-alignment" ]; then
    echo "Cloning face-alignment repository..."
    git clone https://github.com/1adrianb/face-alignment.git
    cd face-alignment
    pip install -e .
    cd ..
else
    echo "face-alignment repository already exists. Skipping clone."
fi

# Clone EmoNet repository
if [ ! -d "emonet" ]; then
    echo "Cloning EmoNet repository..."
    git clone https://github.com/face-analysis/emonet.git
    
    # Create symbolic link in tools directory
    mkdir -p ../tools/
    ln -sf $(pwd)/emonet ../tools/
    
    # Create pretrained directory in EmoNet folder
    mkdir -p emonet/pretrained
    cd emonet/pretrained
    
    # Download pretrained models
    echo "Downloading pretrained EmoNet models..."
    if [ ! -f "emonet_8.pth" ]; then
        wget -O emonet_8.pth https://github.com/face-analysis/emonet/raw/master/pretrained/emonet_8.pth
    fi
    if [ ! -f "emonet_5.pth" ]; then
        wget -O emonet_5.pth https://github.com/face-analysis/emonet/raw/master/pretrained/emonet_5.pth
    fi
    
    cd ../..
else
    echo "EmoNet repository already exists."
    # Ensure the symbolic link exists
    mkdir -p ../tools/
    ln -sf $(pwd)/emonet ../tools/
    
    # Check if model files exist and download if needed
    if [ ! -f "emonet/pretrained/emonet_8.pth" ]; then
        echo "Downloading EmoNet 8-class model..."
        mkdir -p emonet/pretrained
        wget -O emonet/pretrained/emonet_8.pth https://github.com/face-analysis/emonet/raw/master/pretrained/emonet_8.pth
    fi
    if [ ! -f "emonet/pretrained/emonet_5.pth" ]; then
        echo "Downloading EmoNet 5-class model..."
        mkdir -p emonet/pretrained
        wget -O emonet/pretrained/emonet_5.pth https://github.com/face-analysis/emonet/raw/master/pretrained/emonet_5.pth
    fi
fi

cd ..

echo "EmoNet environment setup completed successfully!"
echo "Model capabilities:"
echo "- Emotion classification (8 or 5 classes)"
echo "- Valence and arousal prediction"
echo "- Facial landmarks"
echo ""
echo "The 256-dimensional embeddings come from the last layer before classification."
