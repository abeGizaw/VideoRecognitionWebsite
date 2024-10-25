import os
import pandas as pd
from trainHelper import load_training_data, load_validation_data
from CNN3D import CreateDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np

print('Loading Kinetics-700 dataset...\n')
current_dir = os.path.dirname(os.path.abspath(__file__))
annotations_path = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/annotations')
train_csv = os.path.join(annotations_path, 'train.csv')
val_csv = os.path.join(annotations_path, 'val.csv')
train_videos_dir = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/train')
val_videos_dir = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/val')

# Load CSV files
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

# Load or compute training mappings
print('Loading training video mappings from cache if available...')
train_video_paths, train_video_labels = load_training_data(train_df, train_videos_dir)


# Load or compute validation mappings
print('Loading validation video mappings from cache if available...')
val_video_paths, val_video_labels = load_validation_data(val_df, val_videos_dir)



train_video_labels = np.array(train_video_labels)
val_video_labels = np.array(val_video_labels)

unique_train_labels, train_counts = np.unique(train_video_labels, return_counts=True)
unique_val_labels, val_counts = np.unique(val_video_labels, return_counts=True)
print(f"Unique training labels: {len(unique_train_labels)}")
print(f"Unique validation labels: {len(unique_val_labels)}")

most_common_train_label = unique_train_labels[np.argmax(train_counts)]
most_common_val_label = unique_val_labels[np.argmax(val_counts)]
least_common_train_label = unique_train_labels[np.argmin(train_counts)]
least_common_val_label = unique_val_labels[np.argmin(val_counts)]
print(f"Most common training label: {most_common_train_label} with {np.max(train_counts)} videos.")
print(f"Most common validation label: {most_common_val_label} with {np.max(val_counts)} videos.")
print(f"Least common training label: {least_common_train_label} with {np.min(train_counts)} videos.")
print(f"Least common validation label: {least_common_val_label} with {np.min(val_counts)} videos.")

most_common_train_proportion = np.argmax(train_counts) / len(train_video_labels)
most_common_val_proportion = np.argmax(val_counts) / len(val_video_labels)
print(f"Proportion of most common training label: {most_common_train_proportion:.2f}")
print(f"Proportion of most common validation label: {most_common_val_proportion:.2f}")