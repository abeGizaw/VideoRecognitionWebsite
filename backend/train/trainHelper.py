import os
import pandas as pd
from joblib import Parallel, delayed
import pickle
 
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
    # sife of df
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