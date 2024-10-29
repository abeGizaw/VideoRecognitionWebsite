import os
import pandas as pd
from trainHelper import load_training_data, load_validation_data, createStats

print('Loading Kinetics-700 dataset...')
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
train_video_paths, train_video_labels = load_training_data(train_df, train_videos_dir, "kinetics")

# Load or compute validation mappings
print('Loading validation video mappings from cache if available...')
val_video_paths, val_video_labels = load_validation_data(val_df, val_videos_dir, "kinetics")



print('\nLoading Jester dataset...')
jester_path = os.path.join(current_dir, '../../../data/jester')
jester_path_train = os.path.join(jester_path, 'jester-v1-train.csv') 
jester_path_validation = os.path.join(jester_path, 'jester-v1-validation.csv') 
jester_path_videos =  os.path.join(jester_path,'20bn-jester-v1')

# Load the CSV files
jester_training_df = pd.read_csv(jester_path_train, header=None, names=['id', 'label'], sep=';')
jester_validation_df = pd.read_csv(jester_path_validation, header=None, names=['id', 'label'], sep = ';')

# Load or compute training mappings
print('Loading jester training video mappings from cache if available...')
jesterVideo_train_paths, jester_train_video_labels = load_training_data(jester_training_df,jester_path_videos, "jester")

# Load or compute validation mappings
print('Loading jester validation video mappings from cache if available...')
jesterVideo_val_paths, jester_val_video_labels = load_validation_data(jester_validation_df,jester_path_videos, "jester")


# Kinetics Data
train_video_labels_series = pd.Series(train_video_labels)
val_video_labels_series = pd.Series(val_video_labels)


# Jester Data
jester_train_video_labels = pd.Series(jester_train_video_labels)
jester_val_video_labels = pd.Series(jester_val_video_labels)

# Create Stats
createStats(jester_train_video_labels, "Jester", "training")
createStats(jester_val_video_labels, "Jester", "validation")
createStats(train_video_labels_series, "Kinetics", "training")
createStats(val_video_labels_series, "Kinetics", "validation")
createStats(pd.concat([jester_train_video_labels, train_video_labels_series]), "Jester and Kinetics", "training")
createStats(pd.concat([jester_val_video_labels, val_video_labels_series]), "Jester and Kinetics", "validation")
    