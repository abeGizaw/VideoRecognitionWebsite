import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_video
from torch.utils.data import DataLoader, Dataset

class VideoDataset(Dataset):
    def __init__(self, video_paths, video_labels, num_frames=32, transform=None):
        self.video_paths = video_paths
        self.video_labels = video_labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def _load_images_from_dir(self, dir_path):
        pass;

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_label = self.video_labels[idx]   

        if os.path.isdir(video_path):
            self._load_images_from_dir(video_path); # Skip directories for now (Jester)
            return
    
        # Read video frames using torchvision's read_video (ignoring the audio and info metadata)
        # video = a torch.Tensor containing the video frames in the format (T, H, W, C)
        video, _, _ = read_video(video_path, pts_unit='sec')
        
        # Sample a fixed number of frames evenly from the video
        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            frame_indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            video = video[frame_indices]
        else:
            # If video has fewer frames than required, repeat the last frame
            padding = self.num_frames - total_frames
            video = torch.cat([video, video[-1:].repeat(padding, 1, 1, 1)], dim=0)
        
        # Permute dimensions to (T, C, H, W)
        video = video.permute(0, 3, 1, 2)
        
        # Apply transformations if provided
        if self.transform:
            video = self.transform(video)
        
        return video,video_label

# Function to create a DataLoader for the videos
def create_dataloader(video_paths,video_labels,  num_frames=32, batch_size=5, preprocess=None):
    dataset = VideoDataset(video_paths, video_labels, num_frames=num_frames, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
