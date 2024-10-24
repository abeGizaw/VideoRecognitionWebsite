import os
import pandas as pd
from joblib import Parallel, delayed
import pickle

print('Loading Kinetics-700 dataset...\n')
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

def map_videos_to_labels_joblib(df, video_dir, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(check_video_exists)(row, video_dir) for _, row in df.iterrows()
    )
    video_paths = [res[0] for res in results if res[0] is not None]
    video_labels = [res[1] for res in results if res[1] is not None]
    return video_paths, video_labels

def cache_mappings(file_name, data=None):
    if data is None:  # Load cache
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        return None
    else:  # Save cache
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

# Load or compute training mappings
print('Loading training video mappings from cache if available...')
train_mappings = cache_mappings('train_cache.pkl')
if train_mappings is not None:
    train_video_paths, train_video_labels = train_mappings
    print(f"Loaded {len(train_video_paths)} training videos from cache.\n")
else:
    print('Mapping training videos to labels...')
    train_video_paths, train_video_labels = map_videos_to_labels_joblib(train_df, train_videos_dir)
    print(f"Loaded {len(train_video_paths)} training videos.")
    print(f"Train Video Labels: {len(train_video_labels)}")
    print("Caching training video mappings...\n")
    cache_mappings('train_cache.pkl', (train_video_paths, train_video_labels))

# Load or compute validation mappings
print('Loading validation video mappings from cache if available...')
val_mappings = cache_mappings('val_cache.pkl')
if val_mappings is not None:
    val_video_paths, val_video_labels = val_mappings
    print(f"Loaded {len(val_video_paths)} validation videos from cache.\n")
else:
    print('Mapping validation videos to labels...')
    val_video_paths, val_video_labels = map_videos_to_labels_joblib(val_df, val_videos_dir)
    print(f"Loaded {len(val_video_paths)} validation videos.")
    print(f"Val Video Labels: {len(val_video_labels)}")
    print("Caching validation video mappings...\n")
    cache_mappings('val_cache.pkl', (val_video_paths, val_video_labels))
