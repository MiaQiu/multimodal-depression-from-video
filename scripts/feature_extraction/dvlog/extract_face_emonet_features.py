import os
import cv2
import torch
import joblib
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms

import sys
sys.path.append('./')
from tools.emonet.emonet.models.emonet import EmoNet

def load_face_batch(faces, target_width=256, target_height=256, transform=transforms.ToTensor()):
    """Process a batch of faces at once"""
    try:
        batch_tensors = []
        for face in faces:
            try:
                image = face.astype(np.uint8)
                assert(image.ndim==3 and image.shape[2]==3)
                assert(image.dtype == np.uint8)
                
                image = cv2.resize(image, (target_width, target_height))
                image = np.ascontiguousarray(image)
                tensor_img = transform(image)
                batch_tensors.append(tensor_img)
            except Exception:
                # Skip problematic faces
                continue
                
        if not batch_tensors:
            return None
            
        # Stack all tensors into a batch
        batch = torch.stack(batch_tensors).to(args.cuda_device)
        return batch
        
    except Exception as e:
        print(f"Error in load_face_batch: {e}")
        return None

# Create a hook to capture the 256-dimensional embeddings
activation = {}
def get_activation(name):
    def hook(model, input, output):
        # For avg_pool_2, we need to reshape the output to a flat 256-dim vector
        if name == 'avg_pool_features':
            batch_size = output.shape[0]
            activation[name] = output.view(batch_size, output.shape[1])
        else:
            activation[name] = output
    return hook

def process_video(facesID):
    try:
        faces_path = os.path.join(args.faces_dir, facesID)
        
        if not os.path.exists(faces_path):
            print(f"Error: Face file not found at {faces_path}")
            return
            
        faces = np.load(faces_path)["data"]

        dst_embedding_path = os.path.join(args.face_embeddings_output_dir, facesID)

        if os.path.exists(dst_embedding_path):
            print(f"Skipping sample {facesID} because it was already processed")
        else:
            embedding_seq = np.empty((0, 256))  # 256-dimensional embeddings
            
            # Process in batches for faster processing
            batch_size = args.batch_size
            total_faces = len(faces)
            
            # Skip frames if needed to speed up processing
            if args.process_every_n_frames > 1:
                faces = faces[::args.process_every_n_frames]
                print(f"Processing every {args.process_every_n_frames} frames: {len(faces)}/{total_faces} faces")
            
            for i in range(0, len(faces), batch_size):
                # Get a batch of faces
                face_batch = faces[i:i+batch_size]
                
                # Only print status occasionally to reduce overhead
                if i % (batch_size * 10) == 0:
                    print(f"Processing batch {i//batch_size + 1}/{(len(faces) + batch_size - 1)//batch_size}")
                
                # Process the batch
                face_tensors = load_face_batch(face_batch)
                if face_tensors is None:
                    continue
                    
                try:
                    # Forward pass through EmoNet
                    with torch.no_grad():
                        # Clear previous activations
                        if 'avg_pool_features' in activation:
                            del activation['avg_pool_features']
                            
                        # Run the model
                        _ = emonet(face_tensors)
                        
                        # Get the embedding from our hook (256-dim embedding from last layer before classification)
                        if 'avg_pool_features' in activation:
                            embedding = activation['avg_pool_features'].cpu().numpy()
                            embedding_seq = np.vstack((embedding_seq, embedding))
                        else:
                            print(f"Warning: No activation captured for batch {i//batch_size + 1}")
                except Exception as e:
                    print(f"Error in EmoNet inference for batch {i//batch_size + 1}: {e}")

            if len(embedding_seq) > 0:
                print(f"Saving {len(embedding_seq)} embeddings to {dst_embedding_path}")
                np.savez_compressed(dst_embedding_path, data=embedding_seq)
            else:
                print(f"No embeddings generated for {facesID}")
    except Exception as e:
        print(f"Error in process_video for {facesID}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda")
    parser.add_argument("--n-expression", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="./feature_extractors/emonet/pretrained/emonet_8.pth")
    parser.add_argument("--faces-dir", type=str, default="./data/D-vlog/no-chunked/faces/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--process-every-n-frames", type=int, default=1) 
    parser.add_argument("--left-index", type=int, default=0)
    parser.add_argument("--right-index", type=int, default=961)
    parser.add_argument("--face-embeddings-output-dir", required=True, type=str)
    args = parser.parse_args()

    emotion_map = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}

    print(f"Loading EmoNet model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    print(f"Creating EmoNet model with {args.n_expression} expressions")
    emonet = EmoNet(n_expression=args.n_expression).to(args.cuda_device)
    
    # Register a forward hook to capture the 256-dimensional embeddings
    # This captures the output of avg_pool_2, which is the last layer before classification
    emonet.avg_pool_2.register_forward_hook(get_activation('avg_pool_features'))
    
    emonet.load_state_dict(checkpoint)
    emonet.eval()

    os.makedirs(args.face_embeddings_output_dir, exist_ok=True)

    print(f"Looking for face files in {args.faces_dir}")
    facesIDs = [video for video in sorted(os.listdir(args.faces_dir))][args.left_index:args.right_index]
    print(f"Found {len(facesIDs)} face files")
    
    # Process files in parallel
    loop = tqdm(facesIDs)
    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(process_video)(facesID) for facesID in loop
    )
