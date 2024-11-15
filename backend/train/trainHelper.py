import os
import pandas as pd
from joblib import Parallel, delayed
import pickle
import torch
from trainingMappings import combineLabels, generalized
 
def get_kinetics_dataFrames(*, combine_labels = False, generalized_data = False):
    print('Loading Kinetics-700 dataset...')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    annotations_path = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/annotations')
    train_csv = os.path.join(annotations_path, 'train.csv')
    val_csv = os.path.join(annotations_path, 'val.csv')
    train_videos_dir = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/train')
    val_videos_dir = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/val')
    
    if combine_labels:
        combineLabels(train_videos_dir, 500, 'train')
        combineLabels(val_videos_dir, 50, 'val')

    # Load CSV files
    train_df = pd.read_csv(train_csv)

 
    val_df = pd.read_csv(val_csv)
    train_video_paths, train_video_labels = load_training_data(train_df, train_videos_dir, "kinetics")
    val_video_paths, val_video_labels = load_validation_data(val_df, val_videos_dir, "kinetics")
    
    
    
    # Kinetics Data
    train_video_labels_series = pd.Series(train_video_labels)
    val_video_labels_series = pd.Series(val_video_labels)
    

    # Create Stats
    # createStats(train_video_labels_series, "Kinetics", "training")
    # createStats(val_video_labels_series, "Kinetics", "validation")
    # createStats(pd.concat([jester_train_video_labels, train_video_labels_series]), "Jester and Kinetics", "training")
    # createStats(pd.concat([jester_val_video_labels, val_video_labels_series]), "Jester and Kinetics", "validation")

    kinetics_test_df = pd.DataFrame({
        'video_path': val_video_paths,
        'label': val_video_labels
    })

    kinetics_train_df = pd.DataFrame({
        'video_path': train_video_paths,
        'label': train_video_labels
    })


    if generalized_data:
        kinetics_test_df = pd.concat([kinetics_test_df, addGeneralized(val_videos_dir)], ignore_index=True)
        kinetics_train_df = pd.concat([kinetics_train_df, addGeneralized(train_videos_dir)], ignore_index=True)


    return kinetics_train_df, kinetics_test_df



def addGeneralized(dir_path):
    video_paths = []
    labels = []

    # Iterate over each label in the generalized list
    for label in generalized:
        label_dir = os.path.join(dir_path, label)        
        if os.path.isdir(label_dir):
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                if os.path.isfile(file_path):
                    video_paths.append(file_path)
                    labels.append(label)
    
    generalized_df = pd.DataFrame({
        'video_path': video_paths,
        'label': labels
    })
    
    return generalized_df


def load_training_data(train_df, train_videos_dir, cache_name):
    train_mappings = cache_mappings(f'{cache_name}_train.pkl')
    if train_mappings is not None:
        train_video_paths, train_video_labels = train_mappings
        print(f"Loaded {len(train_video_paths)} training videos from cache.\n")
    else:
        print('Mapping training videos to labels...')
        train_video_paths, train_video_labels = map_videos_to_labels_joblib(train_df, train_videos_dir, cache_name=cache_name)
        print(f"Loaded {len(train_video_paths)} training videos.")
        print(f"Train Video Labels: {len(train_video_labels)}")
        print("Caching training video mappings...\n")
        cache_mappings(f'{cache_name}_train.pkl', (train_video_paths, train_video_labels))
 
    return train_video_paths, train_video_labels
 
 
def load_validation_data(val_df, val_videos_dir, cache_name):
    val_mappings = cache_mappings(f'{cache_name}_val.pkl')
    if val_mappings is not None:
        val_video_paths, val_video_labels = val_mappings
        print(f"Loaded {len(val_video_paths)} validation videos from cache.\n")
    else:
        print('Mapping validation videos to labels...')
        val_video_paths, val_video_labels = map_videos_to_labels_joblib(val_df, val_videos_dir, cache_name = cache_name)
        print(f"Loaded {len(val_video_paths)} validation videos.")
        print(f"Val Video Labels: {len(val_video_labels)}")
        print("Caching validation video mappings...\n")
        cache_mappings(f'{cache_name}_val.pkl', (val_video_paths, val_video_labels))
 
    return val_video_paths, val_video_labels
 
 
def check_video_exists(row, video_dir, cache_name):
    if cache_name == 'jester':
        label = row['label']
        folder_name = row['id']
        folder_path = os.path.join(video_dir, str(folder_name))
        if os.path.exists(folder_path):
            return folder_path, label
        else: 
            print(f"Warning: Folder {folder_path} not found in {video_dir} folder.")
    else: 
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
 
def map_videos_to_labels_joblib(df, video_dir, n_jobs=-1,*, cache_name=None):
    results = Parallel(n_jobs=n_jobs)(
        delayed(check_video_exists)(row, video_dir, cache_name) for _, row in df.iterrows()
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


def createStats(df, name, type_of_data="training"):
    """
    Generates and prints statistics for a given dataset.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the dataset.
    name (str): The name of the dataset.
    type_of_data (str, optional): The type of data (e.g., "training", "validation"). Default is "training".

    This function calculates and prints the following statistics for the given dataset:
    - The number of unique labels.
    - The most common label and its count.
    - The least common label and its count.
    - The proportion of the most common label.

    """
    df_val_counts = df.value_counts()
    uniqueLabels = df_val_counts.index        
    print(f'Unique labels in {name} dataset: {len(uniqueLabels)} ') 
    most_common_label = df_val_counts.idxmax()
    least_common_label = df_val_counts.idxmin()
    most_common_count = df_val_counts.max()
    least_common_count = df_val_counts.min()
    most_common_proportion = most_common_count / len(df)
    print(f"{name} most common {type_of_data} label: {most_common_label} with {most_common_count} videos out of {len(df)}.")
    print(f"{name} least common {type_of_data} label: {least_common_label} with {least_common_count} videos out of {len(df)}.")
    print(f"{name} proportion of most common {type_of_data} label: {most_common_proportion} \n")
