"""
CAN ONLY RUN THIS FILE ON ROSE SERVER
"""
import os
import pandas as pd
from trainHelper import load_training_data, load_validation_data, createStats
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from videoCreator import create_dataloader
from trainingMappings import index_to_label_k400, unwanted_labels, new_classes, label_to_index_k400
import torch
 
 
print('Loading Kinetics-700 dataset...')
current_dir = os.path.dirname(os.path.abspath(__file__))
annotations_path = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/annotations')
train_csv = os.path.join(annotations_path, 'train.csv')
val_csv = os.path.join(annotations_path, 'val.csv')
train_videos_dir = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/train')
val_videos_dir = os.path.join(current_dir, '../../../data/kinetics-dataset/k700-2020/val')
 
# Load CSV files
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
train_video_paths, train_video_labels = load_training_data(train_df, train_videos_dir, "kinetics")
val_video_paths, val_video_labels = load_validation_data(val_df, val_videos_dir, "kinetics")
 
 
 
print('\nLoading Jester dataset...')
jester_path = os.path.join(current_dir, '../../../data/jester')
jester_path_train = os.path.join(jester_path, 'jester-v1-train.csv')
jester_path_validation = os.path.join(jester_path, 'jester-v1-validation.csv')
jester_path_videos =  os.path.join(jester_path,'20bn-jester-v1')
 
# Load jester CSV files
jester_training_df = pd.read_csv(jester_path_train, header=None, names=['id', 'label'], sep=';')
jester_validation_df = pd.read_csv(jester_path_validation, header=None, names=['id', 'label'], sep = ';')
jesterVideo_train_paths, jester_train_video_labels = load_training_data(jester_training_df,jester_path_videos, "jester")
jesterVideo_val_paths, jester_val_video_labels = load_validation_data(jester_validation_df,jester_path_videos, "jester")
 
 
# Kinetics Data
train_video_labels_series = pd.Series(train_video_labels)
val_video_labels_series = pd.Series(val_video_labels)
 
# Jester Data
jester_train_video_labels = pd.Series(jester_train_video_labels)
jester_val_video_labels = pd.Series(jester_val_video_labels)
 
 
# Create Stats
# createStats(jester_train_video_labels, "Jester", "training")
# createStats(jester_val_video_labels, "Jester", "validation")
# createStats(train_video_labels_series, "Kinetics", "training")
# createStats(val_video_labels_series, "Kinetics", "validation")
# createStats(pd.concat([jester_train_video_labels, train_video_labels_series]), "Jester and Kinetics", "training")
# createStats(pd.concat([jester_val_video_labels, val_video_labels_series]), "Jester and Kinetics", "validation")
 
jester_train_df = pd.DataFrame({
    'video_path': jesterVideo_train_paths,
    'label': jester_train_video_labels
})
jester_train_df = jester_train_df[
    ~jester_train_df['label'].isin(unwanted_labels)
]
 
jester_test_df = pd.DataFrame({
    'video_path': jesterVideo_val_paths,
    'label': jester_val_video_labels
})
jester_test_df = jester_test_df[
    ~jester_test_df['label'].isin(unwanted_labels)
]
   
kinetics_train_df = pd.DataFrame({
    'video_path': train_video_paths,
    'label': train_video_labels
})
 
kinetics_400_labels = set(index_to_label_k400.values())
 
kinetics_train_filtered_df = kinetics_train_df[
    ~kinetics_train_df['label'].isin(kinetics_400_labels)
]
 
kinetics_test_df = pd.DataFrame({
    'video_path': val_video_paths,
    'label': val_video_labels
})

 
kinetics_test_filtered_df = kinetics_test_df[
    ~kinetics_test_df['label'].isin(kinetics_400_labels)
]
kinetics_test_base_df = kinetics_test_df[
    kinetics_test_df['label'].isin(kinetics_400_labels)
]

mock_test_data = kinetics_test_base_df.iloc[50:60].reset_index(drop=True)
mock_test_data['label_index'] = mock_test_data['label'].map(label_to_index_k400)
print('mock test data', mock_test_data['label_index'])
print('\n')

# Display the shapes to confirm the filtering
print("Filtered Kinetics Train DataFrame Shape:", kinetics_train_filtered_df.shape, "vs Original:", kinetics_train_df.shape)
print("Filtered Kinetics Test DataFrame Shape:", kinetics_test_filtered_df.shape, "vs Original:", kinetics_test_df.shape)
 
 
# kinetics_top2_labels = kinetics_train_df['label'].value_counts().nlargest(2).index
# kinetics_top2_df = kinetics_train_df[kinetics_train_df['label'].isin(kinetics_top2_labels)]
kinetics_strings_df = kinetics_train_df[kinetics_train_df['label'].isin(new_classes)]
mock_train_data = kinetics_strings_df.reset_index(drop=True)
#mock_train_data = pd.concat([kinetics_top2_df, kinetics_gangman_df], ignore_index=True)
# mock_train_data = pd.concat([jester_top2_df, kinetics_top2_df, kinetics_hangman_df], ignore_index=True)
 
print('mock labels being used ', mock_train_data['label'].value_counts())
# print('mock data', mock_train_data.head())
# print('test data' ,mock_test_data)
 
# Load the model with the KINETICS400_IMAGENET22K_V1 pre-trained weights
weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
model = swin3d_b(weights=weights)
preprocess = weights.transforms()
 
"""
MOCKING TRAIN DATA
"""
# Create a dataloader for the mock training data
# dataloader = create_dataloader(mock_train_data['video_path'], mock_train_data['label'].index, num_frames=16, batch_size=50, preprocess=preprocess)
 
 
# number_features = model.head.in_features
# model.head = torch.nn.Linear(number_features, number_features + len(strings))
# # Freeze the model parameters
# for param in model.parameters():
#     param.requires_grad = False
# # Train final layer
# for param in model.head.parameters():
#     param.requires_grad = True
 
 
# # Train the model and optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()
 
 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_epochs = 1
# running_loss = 0.0
# for epoch in range(num_epochs):
#     model.train()
#     for i, batch in enumerate(dataloader):
#         inputs, labels = batch
#         inputs = inputs.to(device)
#         labels = labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(batch)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
 
#     epoch_loss = running_loss / len(dataloader.dataset)
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
 
# Save the model
model_path = os.path.join(current_dir, '../models/trained_swin_model.pth')
torch.save(model.state_dict(), model_path)
 
# for i, batch in enumerate(dataloader):
#     print(f"Batch {i+1} shape: {batch.shape}")
 

 
 
"""
MOCKING MODEL ON TEST DATA
"""
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)  
model.to(device)
dataloader = create_dataloader(
    video_paths=mock_test_data['video_path'],
    video_labels=mock_test_data['label_index'], 
    num_frames=16,
    batch_size=5,
    preprocess=preprocess
) 
 
correct_predictions = 0
top5_correct_predictions = 0
total_predictions = 0
wrong_paths = []

with open("results.txt", "w") as f:
    with torch.no_grad():
        for i, (batch, label) in enumerate(dataloader):
            batch = batch.to(device)  
            label = label.to(device)

            # Get model predictions
            outputs = model(batch)
            predicted_labels = torch.argmax(outputs, dim=1)
            top_5 = torch.topk(outputs, 5).indices
        
            # Convert predictions and labels to lists of indices for comparison
            predicted_labels = predicted_labels.cpu().tolist()
            labels = label.cpu().tolist()
            top_5 = top_5.cpu().tolist()          

            # Convert indices to labels
            predicted_labels_mapped = [index_to_label_k400.get(idx, f"unknown_{idx}") for idx in predicted_labels]
            top_5_mapped = [[index_to_label_k400.get(idx, f"unknown_{idx}") for idx in indices] for indices in top_5]           
            true_labels_mapped = [index_to_label_k400.get(label, label) for label in labels]

            # Calculate accuracy
            correct_predictions += sum(pred == true for pred, true in zip(predicted_labels, labels))
            top5_correct_predictions += sum(true in top_5[i] for i, true in enumerate(labels))
            total_predictions += len(labels)

            print(predicted_labels)
            print(labels)
            print(top_5)


            # Track the paths of videos with incorrect top-5 predictions
            # Track the paths of videos with incorrect top-5 predictions
            batch_start = i * dataloader.batch_size
            batch_video_paths = list(dataloader.dataset.video_paths[batch_start : batch_start + len(labels)])
            
            # Print for debugging purposes
            for j, true_label in enumerate(labels):
                if j < len(batch_video_paths) and true_label not in top_5[j]:
                    wrong_paths.append((batch_video_paths[j], top_5_mapped[j])) 


            f.write(f"Batch {i+1} predictions: {predicted_labels_mapped}\n")
            f.write(f"Batch {i+1} true labels: {true_labels_mapped}\n")
            f.write(f"Batch {i+1} top 5 predictions:\n")
            for entry in top_5_mapped:
                f.write(f"{entry}\n")
            f.write("\n")
           
            

accuracy = correct_predictions / total_predictions
top5_accuracy = top5_correct_predictions / total_predictions
print(f"Accuracy: {accuracy:.4f}")
print(f"Top-5 Accuracy: {top5_accuracy:.4f}")

with open("wrong_paths.txt", "w") as wrong_path:
    for (path, top_5) in wrong_paths:
        wrong_path.write(f"\n {path} \n {top_5}\n")