
import torch
from torch.utils.data import Dataset

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