import os
import pandas as pd
from trainHelper import load_training_data, load_validation_data

print('Loading Kinetics-700 dataset...\n')
current_dir = os.path.dirname(os.path.abspath(__file__))
annotations_path = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/annotations')
train_csv = os.path.join(annotations_path, 'train.csv')
val_csv = os.path.join(annotations_path, 'val.csv')
train_videos_dir = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/train')
val_videos_dir = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/val')


#Here again 
# Load CSV files
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

print("Loading Jester dataset...\n")
current_dir = os.path.dirname(os.path.abspath(__file__))
jester_path = os.path.join(current_dir, '../../../data/jester')


# Load or compute training mappings
print('Loading training video mappings from cache if available...')
train_video_paths, train_video_labels = load_training_data(train_df, train_videos_dir)

#test
# Load or compute validation mappings
print('Loading validation video mappings from cache if available...')
val_video_paths, val_video_labels = load_validation_data(val_df, val_videos_dir)


train_video_labels_series = pd.Series(train_video_labels)
val_video_labels_series = pd.Series(val_video_labels)

# Returns a Series object with the counts of each unique value in the Series and the unique values
train_label_counts = train_video_labels_series.value_counts()
unique_train_labels = train_label_counts.index
print(f"Unique training labels: {len(unique_train_labels)}")

most_common_train_label = train_label_counts.idxmax()
least_common_train_label = train_label_counts.idxmin()
most_common_train_count = train_label_counts.max()
least_common_train_count = train_label_counts.min()
most_common_train_proportion = most_common_train_count / len(train_video_labels)
print(f"Most common training label: {most_common_train_label} with {most_common_train_count} videos out of {len(train_video_labels_series)}.")
print(f"Least common training label: {least_common_train_label} with {least_common_train_count} videos out of {len(train_video_labels_series)}.")
print(f"Proportion of most common training label: {most_common_train_proportion} \n")

val_label_counts = val_video_labels_series.value_counts()
unique_val_labels = val_label_counts.index
print(f"Unique validation labels: {len(unique_val_labels)}")

most_common_val_label = val_label_counts.idxmax()
least_common_val_label = val_label_counts.idxmin()
most_common_val_count = val_label_counts.max()
least_common_val_count = val_label_counts.min()
most_common_val_proportion = most_common_val_count / len(val_video_labels)
print(f"Most common validation label: {most_common_val_label} with {most_common_val_count} videos out of {len(val_video_labels_series)}.")
print(f"Least common validation label: {least_common_val_label} with {least_common_val_count} videos out of {len(val_video_labels_series)}.")
print(f"Proportion of most common validation label: {most_common_val_proportion} \n")



