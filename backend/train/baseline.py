"""
CAN ONLY RUN THIS FILE ON ROSE SERVER
"""
import os
import time
import pandas as pd
from trainHelper import get_kinetics_dataFrames
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from videoCreator import create_dataloader
from trainingMappings import index_to_label_k400, unwanted_labels, new_classes, label_to_index_k400,combineLabels,generalized
import torch
import torch.nn.functional as F
from torch import optim 

kinetics_train_df, kinetics_test_df = get_kinetics_dataFrames()


kinetics_400_labels = set(index_to_label_k400.values())
 
kinetics_train_filtered_df = kinetics_train_df[
    ~kinetics_train_df['label'].isin(kinetics_400_labels)
]
 
kinetics_test_filtered_df = kinetics_test_df[
    ~kinetics_test_df['label'].isin(kinetics_400_labels)
]

# Display the shapes to confirm the filtering
print("Filtered Kinetics Train DataFrame Shape:", kinetics_train_filtered_df.shape, "vs Original:", kinetics_train_df.shape)
print("Filtered Kinetics Test DataFrame Shape:", kinetics_test_filtered_df.shape, "vs Original:", kinetics_test_df.shape)
print('\n')
 

 
"""
MOCKING TRAIN DATA
"""

kinetics_train_base_df = kinetics_train_df[
    kinetics_train_df['label'].isin(kinetics_400_labels)
]
 

mock_train_data =  kinetics_train_base_df.reset_index(drop=True)
# mock_train_data =  kinetics_strings_df.head(200).reset_index(drop=True)
mock_train_data['label_index'] = mock_train_data['label'].map(label_to_index_k400)
print('mock train labels being used \n', mock_train_data['label'].value_counts())
print('train label size is: ', mock_train_data.shape)



"""
MOCKING MODEL ON TEST DATA
"""

target_labels = ["adjusting glasses"] 

kinetics_test_base_df = kinetics_test_df[
    kinetics_test_df['label'].isin(kinetics_400_labels)
]
filtered_test_df = kinetics_test_df[
    kinetics_test_df['label'].isin(target_labels)
]



mock_test_data = kinetics_test_base_df.reset_index(drop=True)
mock_test_data['label_index'] = mock_test_data['label'].map(label_to_index_k400)

correct_predictions = 0
top5_correct_predictions = 0
total_predictions = 0


# Get the 5 most common labels in the training data
top_5_most_common_labels = kinetics_train_df['label'].value_counts().head(5)
print("Most common labels in training data:\n", top_5_most_common_labels)
total_top5_label_occurrences = top_5_most_common_labels.sum()

top_most_common_labels = kinetics_train_df['label'].value_counts().head(1)
print("Most common labels in training data:\n", top_most_common_labels)
total_top1_label_occurrences = top_most_common_labels.sum()

print("Total occurrences of the top 5 labels:", total_top5_label_occurrences)
print("Total number of labels in training data:", len(kinetics_train_df))
         
           

accuracy = (total_top1_label_occurrences / len(kinetics_train_df))*100
top5_accuracy = (total_top5_label_occurrences / len(kinetics_train_df))*100
print(f"Accuracy: {accuracy:.4f}")
print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
