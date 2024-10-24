import os
import pickle
from joblib import Parallel, delayed

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