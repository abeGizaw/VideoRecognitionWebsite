import torch
import os
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from train.trainingMappings import index_to_label_k400, index_to_label_k400_generalized
from train.videoCreator import create_dataloader
import torch.nn.functional as F
import requests

def process_video(file_path):
    """
    Takes in the video path, processes it with the CNN model, 
    and returns the result.
    """
    print(f"Processing video at {file_path}")
    result = ""
    mostConfident = ""
    usedParallel = False

    # Load the pre-trained weights
    weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    model = swin3d_b(weights=weights)
    preprocess = weights.transforms()


    # Modify the final layer to output correct number of classes. 
    # Change to 401 if using base model
    num_features = model.head.in_features
    model.head = torch.nn.Linear(num_features, 400)
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False

    # Only allow gradients on the final layer
    for param in model.head.parameters():
        param.requires_grad = True

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # Use all available GPUs
        usedParallel = True
    model.to(device)

    # Construct the absolute path to the model weights file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # If using base model, use this path and everywhere that uses generalized model replace with this path / comment out the generalized model path
    # base_model_path = os.path.join(current_dir, 'trained_swin_model_base.pth')
    gen_model_path = os.path.join(current_dir, 'trained_swin_model_generalized.pth')
    if not os.path.exists(gen_model_path):
        # Pass in base as model type if using base model
        download_file_from_github_release(gen_model_path, "generalized")

   
    state_dict = torch.load(gen_model_path, map_location=device, weights_only=True)
    if not usedParallel:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Create a DataLoader for the single video
    dataloader = create_dataloader(
        video_paths=[file_path],     
        video_labels=None,           
        num_frames=16,
        batch_size=1,
        preprocess=preprocess
    )

   
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
            label = index_to_label_k400_generalized.get(predicted_label, "Unknown")
            top_5_labels = [index_to_label_k400_generalized.get(i.item(), "Unknown") for i in top_5_indices]

            # Build the result string
            result = f"Most confident: {label}\nI think it is at least one of these 5:\n"
            mostConfident = label
            for lbl, confidence in zip(top_5_labels, top_5_confidences):
                result += f"{confidence.item() * 100:.2f}%: {lbl}\n"
                
        return result, mostConfident
    

def download_file_from_github_release(destination, model_type):
    """
    Downloads the model weights file from a GitHub release.
    This function writes the file to the specified destination.
    """
    url = f"https://github.com/abeGizaw/VideoRecognitionWebsite/releases/download/v1.0.0/trained_swin_model_{model_type}.pth"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"File downloaded successfully to {destination}")
    else:
        raise Exception(f"Failed to download file. HTTP Status Code: {response.status_code}")
