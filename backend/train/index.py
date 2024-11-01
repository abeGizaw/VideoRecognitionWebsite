"""
CAN ONLY RUN THIS FILE ON ROSE SERVER
"""
import os
import pandas as pd
from trainHelper import load_training_data, load_validation_data, createStats
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from videoCreator import create_dataloader
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
createStats(jester_train_video_labels, "Jester", "training")
createStats(jester_val_video_labels, "Jester", "validation")
createStats(train_video_labels_series, "Kinetics", "training")
createStats(val_video_labels_series, "Kinetics", "validation")
createStats(pd.concat([jester_train_video_labels, train_video_labels_series]), "Jester and Kinetics", "training")
createStats(pd.concat([jester_val_video_labels, val_video_labels_series]), "Jester and Kinetics", "validation")

jester_train_df = pd.DataFrame({
    'video_path': jesterVideo_train_paths, 
    'label': jester_train_video_labels
})
jester_train_df = jester_train_df[
    ~jester_train_df['label'].isin(["Doing other things", "no gesture"])
]

jester_test_df = pd.DataFrame({
    'video_path': jesterVideo_val_paths,
    'label': jester_val_video_labels
})
jester_test_df = jester_test_df[
    ~jester_test_df['label'].isin(["Doing other things", "no gesture"])
]
    
kinetics_train_df = pd.DataFrame({
    'video_path': train_video_paths,
    'label': train_video_labels
})

kinetics_test_df = pd.DataFrame({
    'video_path': val_video_paths,
    'label': val_video_labels
})

mock_test_data = kinetics_test_df.head()

# jester_top2_labels = jester_train_df['label'].value_counts().nlargest(2).index
# jester_top2_df = jester_train_df[jester_train_df['label'].isin(jester_top2_labels)][:2000]
kinetics_top2_labels = kinetics_train_df['label'].value_counts().nlargest(2).index
kinetics_top2_df = kinetics_train_df[kinetics_train_df['label'].isin(kinetics_top2_labels)]
kinetics_hangman_df = kinetics_train_df[kinetics_train_df['label'].str.contains("dancing gangnam style", case=False, na=False)]
mock_train_data = pd.concat([kinetics_top2_df, kinetics_hangman_df], ignore_index=True)
# mock_train_data = pd.concat([jester_top2_df, kinetics_top2_df, kinetics_hangman_df], ignore_index=True)

print('mock labels being used ', mock_train_data['label'].value_counts())

print(mock_train_data.head())
print(mock_train_data.shape)


# Load the model with the KINETICS400_IMAGENET22K_V1 pre-trained weights
weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
model = swin3d_b(weights=weights)
preprocess = weights.transforms()

"""
MOCKING TRAIN DATA
"""
# dataloader = create_dataloader(mock_train_data['video_path'], num_frames=16, batch_size=50, preprocess=preprocess)

# for i, batch in enumerate(dataloader):
#     print(f"Batch {i+1} shape: {batch.shape}")


"""
MOCKING MODEL ON TEST DATA
"""
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)  
model.to(device)
dataloader = create_dataloader(mock_test_data['video_path'], num_frames=16, batch_size=5, preprocess=preprocess)
for i, batch in enumerate(dataloader):
    print(f"Batch {i+1} shape: {batch.shape}")
