import os
import pandas as pd
from trainHelper import load_training_data, load_validation_data

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
train_video_paths, train_video_labels = load_training_data(train_df, train_videos_dir)


# Load or compute validation mappings
print('Loading validation video mappings from cache if available...')
val_video_paths, val_video_labels = load_validation_data(val_df, val_videos_dir)
