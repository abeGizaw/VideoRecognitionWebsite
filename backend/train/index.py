"""
CAN ONLY RUN THIS FILE ON ROSE SERVER
"""
import os
import time
from trainHelper import get_kinetics_dataFrames
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from videoCreator import create_dataloader
from trainingMappings import label_to_index_k400_generalized, index_to_label_k400_generalized
import torch
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
 

kinetics_train_df, kinetics_test_df = get_kinetics_dataFrames(generalized_data=True)

generalized_kinetics_400_labels = set(index_to_label_k400_generalized.values())
 
# Filter out the Kinetics-400 labels from the data (Didn't end up using all of them) 
kinetics_train_filtered_df = kinetics_train_df[
    ~kinetics_train_df['label'].isin(generalized_kinetics_400_labels)
]
 
kinetics_test_filtered_df = kinetics_test_df[
    ~kinetics_test_df['label'].isin(generalized_kinetics_400_labels)
]

# Display the shapes to confirm the filtering
print("Filtered Kinetics Train DataFrame Shape:", kinetics_train_filtered_df.shape, "vs Original:", kinetics_train_df.shape)
print("Filtered Kinetics Test DataFrame Shape:", kinetics_test_filtered_df.shape, "vs Original:", kinetics_test_df.shape)
print('\n')
 
 
# Load the model with the KINETICS400_IMAGENET22K_V1 pre-trained weights
weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
model = swin3d_b(weights=weights)
preprocess = weights.transforms()
 
"""
TRAINING DATA
"""

num_features = model.head.in_features
kinetics_train_base_df = kinetics_train_df[
    kinetics_train_df['label'].isin(generalized_kinetics_400_labels)
].reset_index(drop=True)

# Modify the final layer to output the proper amount of classes
model.head = torch.nn.Linear(num_features, kinetics_train_base_df['label'].nunique())

# Freeze all layers except the final layer
for param in model.parameters():
    param.requires_grad = False

# Only allow gradients on the final layer
for param in model.head.parameters():
    param.requires_grad = True



# mock_train_data =  kinetics_strings_df.head(200).reset_index(drop=True)
kinetics_train_base_df['label_index'] = kinetics_train_base_df['label'].map(label_to_index_k400_generalized)

print('train label size is: ', kinetics_train_base_df.shape)
print('Number of Unique Labels: ', kinetics_train_base_df['label_index'].nunique())



# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Wrap the model with DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    model = torch.nn.DataParallel(model)  # Use all available GPUs
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use the first 4 GPUs
else:
    print("Using a single GPU or CPU for training")

# Load the model weights
current_dir = os.path.dirname(os.path.abspath(__file__))
model.to(device)
model_path = os.path.join(current_dir, '../models/trained_swin_model_generalized.pth')
state_dict = torch.load(model_path, map_location=device,weights_only=True)
model.load_state_dict(state_dict, strict=True)


dataloader = create_dataloader(
    video_paths=kinetics_train_base_df['video_path'], 
    video_labels=kinetics_train_base_df['label_index'],  
    num_frames=16,
    batch_size=64,
    preprocess=preprocess
)


# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.module.head.parameters() if isinstance(model, torch.nn.DataParallel) else model.head.parameters(), 
    lr=0.001, 
    momentum=0.9
)
print("starting training \n")

# Fine-tune the model
training_losses = []
num_epochs = 1  
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, (inputs, labels) in enumerate(dataloader):
        if inputs is None or labels is None:
            continue

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        training_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        if i % 10 == 0:
            print(f'Batch {i} Loss: {loss.item():.4f}')
            print(f'Time: {time.time() - start_time:.4f} seconds')
        # Save the model every 100 batches
        if i % 100 == 0:
            print("saving model with loss: ", loss.item())
            torch.save(model.state_dict(), model_path)


    end_time = time.time()
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch {epoch + 1} Time: {end_time - start_time:.4f} seconds')
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Save the model after each epoch
    torch.save(model.state_dict(), model_path)

# Save the training losses to a file
with open("training_losses_gen.txt", "w") as f:
    f.write(f"{training_losses}\n")

# Plot the training losses
plt.plot(training_losses)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss_e2.png')
plt.show()

"""
MODEL ON TEST DATA
"""

# Use this list to test on certain classes
target_labels = ["adjusting glasses"] 

kinetics_test_base_df = kinetics_test_df[
    kinetics_test_df['label'].isin(generalized_kinetics_400_labels)
].reset_index(drop=True)

filtered_test_df = kinetics_test_df[
    kinetics_test_df['label'].isin(target_labels)
]

# Limit the test data to 100 samples per class
limited_test_df = filtered_test_df.groupby('label').head(100).reset_index(drop=True)
limited_test_df['label_index'] = limited_test_df['label'].map(label_to_index_k400_generalized)
kinetics_test_base_df['label_index'] = kinetics_test_base_df['label'].map(label_to_index_k400_generalized)

model.eval()
# Load the model weights
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

dataloader = create_dataloader(
    video_paths=kinetics_test_base_df['video_path'],
    video_labels=kinetics_test_base_df['label_index'], 
    num_frames=16,
    batch_size=64,
    preprocess=preprocess
) 
 

correct_predictions = 0
top5_correct_predictions = 0
total_predictions = 0
wrong_paths = []

results = []
# Write all batch results to a file
with open("results_gen.txt", "w") as f:
    with torch.no_grad():
        for i, (batch, label) in enumerate(dataloader):
            batch = batch.to(device)  
            label = label.to(device)

            # Get model predictions
            outputs = model(batch)
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            predicted_label_prob = torch.argmax(probabilities, dim=1)
            predicted_labels = torch.argmax(outputs, dim=1)

            top_5 = torch.topk(outputs, 5).indices
            top_5_confidences = torch.topk(probabilities, 5).values
        
            # Convert predictions and labels to lists of indices for comparison
            predicted_labels = predicted_labels.cpu().tolist()
            predicted_label_prob = predicted_label_prob.cpu().tolist()
            labels = label.cpu().tolist()
            top_5 = top_5.cpu().tolist()   
            top_5_confidences = top_5_confidences.cpu().tolist()       

            # Convert indices to labels
            predicted_labels_mapped = [index_to_label_k400_generalized.get(idx, f"unknown_{idx}") for idx in predicted_labels]
            top_5_mapped = [[index_to_label_k400_generalized.get(idx, f"unknown_{idx}") for idx in indices] for indices in top_5]           
            true_labels_mapped = [index_to_label_k400_generalized.get(label, label) for label in labels]
            label_soft = [index_to_label_k400_generalized.get(idx, "Unknown") for idx in predicted_label_prob]

            # Calculate accuracy
            correct_predictions += sum(pred == true for pred, true in zip(predicted_labels, labels))
            top5_correct_predictions += sum(true in top_5[i] for i, true in enumerate(labels))
            total_predictions += len(labels)

            # print(predicted_labels)
            # print(labels)
            # print(top_5)


            # Track the paths of videos with incorrect top-5 predictions
            batch_start = i * dataloader.batch_size
            batch_video_paths = list(dataloader.dataset.video_paths[batch_start : batch_start + len(labels)])
            
            # Print for debugging purposes
            for j, true_label in enumerate(labels):
                if j < len(batch_video_paths) and true_label not in top_5[j]:
                    wrong_paths.append((batch_video_paths[j], top_5_mapped[j])) 


            f.write(f"Batch {i+1} predictions: {predicted_labels_mapped}\n")
            f.write(f"Batch {i+1} soft predictions: {label_soft}\n")
            f.write(f"Batch {i+1} true labels: {true_labels_mapped}\n")
            f.write(f"Batch {i+1} top 5 predictions:\n")
            for entry in top_5_mapped:
                f.write(f"{entry}\n")
            f.write("\n")

            # # Format the results for display
            # for i, (pred_label, top5_labels, top5_conf) in enumerate(zip(predicted_labels, top_5_mapped, top_5_confidences)):
            #     result = f"Prediction {i + 1}:\n"
            #     result += f"Most confident: {pred_label}\n"
            #     result += "I think it is at least one of these 5:\n"
                
            #     # Format each label and its confidence as a percentage
            #     for lbl, confidence in zip(top5_labels, top5_conf):
            #         result += f"{lbl}: {confidence * 100:.2f}%\n"
                
            #     results.append(result)
           
            
final_output = "\n".join(results)
# print(final_output)
accuracy = correct_predictions / total_predictions
top5_accuracy = top5_correct_predictions / total_predictions
print(f"Accuracy: {accuracy:.4f}")
print(f"Top-5 Accuracy: {top5_accuracy:.4f}")

# Save the wrong paths to a file
with open("wrong_paths_gen.txt", "w") as wrong_path:
    for (path, top_5) in wrong_paths:
        wrong_path.write(f"\n {path} \n {top_5}\n")