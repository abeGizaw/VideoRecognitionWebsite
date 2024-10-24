import os
import pandas as pd
from joblib import Parallel, delayed

annotations_path = './data/kinetics-dataset/k700-2020/annotations'
train_csv = os.path.join(annotations_path, 'train.csv')
val_csv = os.path.join(annotations_path, 'val.csv')

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

train_videos_dir = './data/kinetics-dataset/k700-2020/train'
val_videos_dir = './data/kinetics-dataset/k700-2020/val'


def check_video_exists(row, video_dir):
    label = row['label']
    time_start = str(row['time_start']).zfill(6)
    time_end = str(row['time_end']).zfill(6)
    video_name = f"{row['youtube_id']}_{time_start}_{time_end}.mp4"
    video_path = os.path.join(video_dir, label, video_name)
    
    if os.path.exists(video_path):
        return video_path, label
    else:
        print(f"Warning: Video {video_name} not found in {label} folder.")
        return None, None

# Function to map videos with their labels using joblib for parallel processing
def map_videos_to_labels_joblib(df, video_dir, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(delayed(check_video_exists)(row, video_dir) for _, row in df.iterrows())
    
    video_paths = [res[0] for res in results if res[0] is not None]
    video_labels = [res[1] for res in results if res[1] is not None]
    
    return video_paths, video_labels

train_video_paths, train_video_labels = map_videos_to_labels_joblib(train_df, train_videos_dir)
print(f"Loaded {len(train_video_paths)} training videos.")
print(f"Train Video Labels: {len(train_video_labels)}")

val_video_paths, val_video_labels = map_videos_to_labels_joblib(val_df, val_videos_dir)
print(f"Loaded {len(val_video_paths)} validation videos.")
print(f"Val Video Labels: {len(val_video_labels)}")

