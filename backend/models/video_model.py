import torch
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from train.trainingMappings import index_to_label_k400
from train.videoCreator import create_dataloader
def process_video(file_path):
    """
    Takes in the video path, processes it with the CNN model, 
    and returns the result.
    """
    print(f"Processing video at {file_path}")
    result = ""

    weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    model = swin3d_b(weights=weights)
    model.load_state_dict(torch.load('trained_swin_model.pth'))
    model.eval()
    
    # Preprocessing transforms from the model's weights
    preprocess = weights.transforms()

    # Create a DataLoader for the single video
    dataloader = create_dataloader(
        video_paths=[file_path],     
        video_labels=None,           
        num_frames=32,
        batch_size=1,
        preprocess=preprocess
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            input_video = batch.to(device)

            outputs = model(input_video)
            predicted_label = torch.argmax(outputs, dim=1).item()
            label = index_to_label_k400.get(predicted_label, "Unknown")
            result = f"Action Recognized: {label}"
    
    return result
