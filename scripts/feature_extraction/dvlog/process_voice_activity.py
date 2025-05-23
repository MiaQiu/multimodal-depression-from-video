import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--splits-dir", type=str, default="./data/D-vlog/splits/")
    parser.add_argument("--data-dir", type=str, default="./data/D-vlog/no-chunked/")
    parser.add_argument("--audio-embeddings-dir", type=str, default="audio_pase_embeddings")
    parser.add_argument("--audio-activity-dir", type=str, default="./data/D-vlog/no-chunked/audio_activity/")
    parser.add_argument("--dest-dir", type=str, default="no_voice_idxs")
    args = parser.parse_args()

    # Get only .npz files from the audio embeddings directory
    videoIDs = sorted([f for f in os.listdir(os.path.join(args.data_dir, args.audio_embeddings_dir)) if f.endswith('.npz')])

    train_split = pd.read_csv(f"{args.splits_dir}/training.csv")
    val_split = pd.read_csv(f"{args.splits_dir}/validation.csv")
    test_split = pd.read_csv(f"{args.splits_dir}/test.csv")
    dataset = pd.concat([train_split, val_split, test_split], ignore_index=True)

    os.makedirs(f"{args.data_dir}/{args.dest_dir}/", exist_ok = True)
    for videoID in tqdm(videoIDs):
        # Convert videoID to numeric if it's a string
        video_id_num = int(videoID.split('.')[0]) if isinstance(videoID, str) else videoID
        audio_frame_rate = dataset[dataset["video_id"] == video_id_num]["audio_frame_rate"].values[0]

        audio_path = os.path.join(args.data_dir, args.audio_embeddings_dir, videoID)
        audio_frame_length = np.load(audio_path)['data'].shape[0]

        voice_slots = np.load(os.path.join(args.audio_activity_dir, videoID))["data"]

        no_voice_idxs = []
        for frame_idx in range(audio_frame_length):
            frame_second = frame_idx / audio_frame_rate

            is_voice = False
            for voice_slot in voice_slots:
                if frame_second >= voice_slot[0] and frame_second <= voice_slot[1]:
                    is_voice = True

            if not is_voice:
                no_voice_idxs.append(frame_idx)

        dest_path = f"{args.data_dir}/{args.dest_dir}/{videoID}"
        np.savez_compressed(dest_path, data=np.array(no_voice_idxs))
