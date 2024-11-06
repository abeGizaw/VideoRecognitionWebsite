import torch
import os
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from train.trainingMappings import index_to_label_k400
from train.videoCreator import create_dataloader
import torch.nn.functional as F

def process_video(file_path):
    """
    Takes in the video path, processes it with the CNN model, 
    and returns the result.
    """
    print(f"Processing video at {file_path}")
    result = ""

    weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    model = swin3d_b(weights=weights)
    # Construct the absolute path to the model weights file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'trained_swin_model.pth')
    model.load_state_dict(torch.load(model_path))
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
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities

            predicted_label = torch.argmax(probabilities, dim=1).item()
        
            # Get top 5 probabilities and their corresponding labels
            top_5_indices = torch.topk(probabilities, 5).indices[0]
            top_5_confidences = torch.topk(probabilities, 5).values[0]

            # Convert the predicted label and top 5 labels to human-readable labels
            label = index_to_label_k400.get(predicted_label, "Unknown")
            top_5_labels = [index_to_label_k400.get(i.item(), "Unknown") for i in top_5_indices]

            # Build the result string
            result = f"Most confident: {label}\nI think it is at least one of these 5:\n"
            for lbl, confidence in zip(top_5_labels, top_5_confidences):
                result += f"{lbl}: {confidence.item() * 100:.2f}%\n"
                
        return result
