import os
import pandas as pd
from joblib import Parallel, delayed
import pickle
from trainHelpers import check_video_exists, map_videos_to_labels_joblib, cache_mappings

print('Loading Kinetics-700 dataset...\n')
annotations_path = './data/kinetics-dataset/k700-2020/annotations'
train_csv = os.path.join(annotations_path, 'train.csv')
val_csv = os.path.join(annotations_path, 'val.csv')
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
train_videos_dir = './data/kinetics-dataset/k700-2020/train'
val_videos_dir = './data/kinetics-dataset/k700-2020/val'


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
