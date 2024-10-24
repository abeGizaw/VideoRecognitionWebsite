import os
import pandas as pd
from trainHelper import load_training_data, load_validation_data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np

print('Loading Kinetics-700 dataset...\n')
annotations_path = '../data/kinetics-dataset/k700-2020/annotations'
train_csv = os.path.join(annotations_path, 'train.csv')
val_csv = os.path.join(annotations_path, 'val.csv')
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
train_videos_dir = '../data/kinetics-dataset/k700-2020/train'
val_videos_dir = '../data/kinetics-dataset/k700-2020/val'


# Load or compute training mappings
print('Loading training video mappings from cache if available...')
train_video_paths, train_video_labels = load_training_data(train_df, train_videos_dir)


# Load or compute validation mappings
print('Loading validation video mappings from cache if available...')
val_video_paths, val_video_labels = load_validation_data(val_df, val_videos_dir)


class CreateDataset(Dataset):
    def __init__(self, video_paths, video_labels, transform=None):
        
        self.video_paths = torch.tensor(video_paths, dtype=torch.float32)
        self.video_labels = torch.tensor(video_labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_label = self.video_labels[idx]

        # Load video frames
        video_frames = load_video(video_path)

        if self.transform:
            video_frames = self.transform(video_frames)

        return video_frames, video_label
    
def load_video(video_path):
    # Load video frames
    pass





train_video_labels = pd.array(train_video_labels)
val_video_labels = pd.array(val_video_labels)
train_video_paths = pd.array(train_video_paths)
val_video_paths = pd.array(val_video_paths)

most_common_train = train_video_labels.value_counts().idxmax()
most_common_val = val_video_labels.value_counts().idxmax()
print(f'Most common training label: {most_common_train}')
print(f'Most common validation label: {most_common_val}')