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

# Print the columns of the dataframes
print("Columns in train_df:", train_df.columns)
print("Columns in val_df:", val_df.columns)


# # Load or compute training mappings
# print('Loading training video mappings from cache if available...')
# train_video_paths, train_video_labels = load_training_data(train_df, train_videos_dir, "kinetics")

# #test
# # Load or compute validation mappings
# print('Loading validation video mappings from cache if available...')
# val_video_paths, val_video_labels = load_validation_data(val_df, val_videos_dir, "kinetics")


print("Loading Jester dataset...\n")
current_dir = os.path.dirname(os.path.abspath(__file__))
jester_path = os.path.join(current_dir, '../../../data/jester')
jester_path_train = os.path.join(jester_path, 'jester-v1-train.csv') 
jester_path_validation = os.path.join(jester_path, 'jester-v1-validation.csv') 
jester_path_videos =  os.path.join(jester_path,'20bn-jester-v1')


# Load or compute training mappings with specified column names
jester_training_labels_df = pd.read_csv(jester_path_train, header=None, names=['id', 'label'], sep=';')
# Load validation data with specified column names
jester_validation_labels_df = pd.read_csv(jester_path_validation, header=None, names=['id', 'label'], sep = ';')


jesterVideo_train_paths, jester_train_video_labels = load_training_data(jester_training_labels_df,jester_path_videos, "jester")
jesterVideo_val_paths, jester_val_video_labels = load_validation_data(jester_validation_labels_df,jester_path_videos, "jester")

# #Kinetics Data
# train_video_labels_series = pd.Series(train_video_labels)
# val_video_labels_series = pd.Series(val_video_labels)

# # Returns a Series object with the counts of each unique value in the Series and the unique values
# train_label_counts = train_video_labels_series.value_counts()
# unique_train_labels = train_label_counts.index
# print(f"Unique training labels in Kinetics: {len(unique_train_labels)}")


# # Kinetics things
# most_common_train_label = train_label_counts.idxmax()
# least_common_train_label = train_label_counts.idxmin()
# most_common_train_count = train_label_counts.max()
# least_common_train_count = train_label_counts.min()
# most_common_train_proportion = most_common_train_count / len(train_video_labels)
# print(f"Kinetics most common training label: {most_common_train_label} with {most_common_train_count} videos out of {len(train_video_labels_series)}.")
# print(f"Kinetics least common training label: {least_common_train_label} with {least_common_train_count} videos out of {len(train_video_labels_series)}.")
# print(f"Kinetics proportion of most common training label: {most_common_train_proportion} \n")




# #Kinetics
# val_label_counts = val_video_labels_series.value_counts()
# unique_val_labels = val_label_counts.index
# print(f"Unique validation labels: {len(unique_val_labels)}")

# most_common_val_label = val_label_counts.idxmax()
# least_common_val_label = val_label_counts.idxmin()
# most_common_val_count = val_label_counts.max()
# least_common_val_count = val_label_counts.min()
# most_common_val_proportion = most_common_val_count / len(val_video_labels)
# print(f"Most common validation label: {most_common_val_label} with {most_common_val_count} videos out of {len(val_video_labels_series)}.")
# print(f"Least common validation label: {least_common_val_label} with {least_common_val_count} videos out of {len(val_video_labels_series)}.")
# print(f"Proportion of most common validation label: {most_common_val_proportion} \n")



#Jester 
#Jester Data
jester_train_video_labels = pd.Series(jester_train_video_labels)
jester_val_video_labels = pd.Series(jester_val_video_labels)

# Jester Train
jester_train_label_counts = jester_train_video_labels.value_counts()
jester_unique_train_labels = jester_train_label_counts.index
print(f"Unique training labels in Jester: {len(jester_unique_train_labels)}")
jester_most_common_train_label = jester_train_label_counts.idxmax()
jester_least_common_train_label = jester_train_label_counts.idxmin()
jester_most_common_train_count = jester_train_label_counts.max()
jester_least_common_train_count = jester_train_label_counts.min()
jester_most_common_train_proportion = jester_most_common_train_count / len(jester_train_video_labels)
print(f"Jester most common training label: {jester_most_common_train_label} with {jester_most_common_train_count} videos out of {len(jester_train_label_counts)}.")
print(f"Jester least common training label: {jester_least_common_train_label} with {jester_least_common_train_count} videos out of {len(jester_train_label_counts)}.")
print(f"Jester proportion of most common training label: {jester_most_common_train_proportion} \n")


# Jester Val
jester_val_label_counts = jester_val_video_labels.value_counts()
jester_unique_val_labels = jester_val_label_counts.index
print(f"Unique validation labels in Jester: {len(jester_unique_val_labels)}")


jester_most_common_val_label = jester_val_label_counts.idxmax()
jester_least_common_val_label = jester_val_label_counts.idxmin()
jester_most_common_val_count = jester_val_label_counts.max()
jester_least_common_val_count = jester_val_label_counts.min()
jester_most_common_val_proportion = jester_most_common_val_count / len(jester_val_video_labels)
print(f"Jester most common validation label: {jester_most_common_val_label} with {jester_most_common_val_count} videos out of {len(jester_val_label_counts)}.")
print(f"Jester least common vavalidation label: {jester_least_common_val_label} with {jester_least_common_val_count} videos out of {len(jester_val_label_counts)}.")
print(f"Jester proportion of most common validation label: {jester_most_common_val_proportion} \n")



#Combination
# print(f"Total unique labels: {len(jester_unique_train_labels) +len(unique_train_labels)}")


