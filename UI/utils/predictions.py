import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, types, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.types = torch.tensor(types, dtype=torch.int64)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.types[idx], self.labels[idx]
    
class TrajectoryModel(nn.Module):
    def __init__(self, num_trajectory_types, number_of_features, sequence_length):
        super(TrajectoryModel, self).__init__()
        self.lstm = nn.LSTM(number_of_features, 50, bidirectional=True, batch_first=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Embedding(num_trajectory_types, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 + 2, 100)  # 100 from LSTM + 2 from embedding
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self, x, type_input):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch_size, num_features, seq_len)
        avg_pool_out = self.avg_pool(lstm_out).squeeze(-1)  # (batch_size, 100)
        embedded_type = self.embedding(type_input)
        embedded_type = self.flatten(embedded_type)
        combined = torch.cat((avg_pool_out, embedded_type), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        return x
    
def pad_sequences(sequences, maxlen, dtype='float32', padding='post', value=0):
    padded_sequences = np.full((len(sequences), maxlen, len(sequences[0][0])), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded_sequences[i] = seq[:maxlen]
        else:
            padded_sequences[i, :len(seq)] = seq
    return padded_sequences

def make_predictions(df_test_features, model_name, features, model_path):
    # Extract sequences from the DataFrame
    X = df_test_features.drop(['type_trajectory'], axis=1)
    grouped = X.groupby('id')
    trajectories = [group[features].values for _, group in grouped]
    ids = list(grouped.groups.keys())
    trajectory_type = df_test_features.groupby('id')['type_trajectory'].first().values

    # Check the unique values of trajectory_type
    unique_trajectory_types = np.unique(trajectory_type)
    trajectory_type_to_index = {t: i for i, t in enumerate(unique_trajectory_types)}
    trajectory_type_indices = np.array([trajectory_type_to_index[t] for t in trajectory_type])
    # Pad the sequences
    max_sequence_length = max([len(sequence) for sequence in trajectories])
    trajectories_padded = pad_sequences(trajectories, max_sequence_length, padding='post')

    sequence_length = trajectories_padded.shape[1]
    number_of_features = trajectories_padded.shape[2]

    # Convert the data to PyTorch tensors
    test_sequences_tensor = torch.tensor(trajectories_padded, dtype=torch.float32)
    test_types_tensor = torch.tensor(trajectory_type_indices, dtype=torch.long)

    # Initialize an empty DataFrame for the result
    df_predicted = pd.DataFrame({'id': ids })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the model
    model = TrajectoryModel(len(unique_trajectory_types), number_of_features, sequence_length)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Initialize a list to store predictions
    predictions = []

    test_sequences_tensor = test_sequences_tensor.to(device)
    test_types_tensor = test_types_tensor.to(device)

    # Make predictions using the model
    with torch.no_grad():
        outputs = model(test_sequences_tensor, test_types_tensor)
        predictions.extend(outputs.squeeze().tolist())

    # Convert each label's predictions into a DataFrame
    column_name = model_name
    df_temp = pd.DataFrame({column_name: predictions, 'id': ids})

    # Merge the temporary DataFrame with the main annotated DataFrame
    if df_predicted.empty:
        df_predicted = df_temp
    else:
        df_predicted = df_predicted.merge(df_temp, on='id', how='outer')

    # Drop rows with missing values
    df_predicted.dropna(inplace=True)

    return df_predicted

def edit_predictions(df_predicted, df_test_features, col_name):
    # Merge df_predicted with df_test_features to add recording_day, file_id, and video_in_file_id
    df_test_features_copy = df_test_features.copy()
    df_predicted = df_predicted.merge(df_test_features_copy[['recording_day', 'file_number', 'trial_number_in_file','id']], on='id', how='left')
    df_predicted = df_predicted.drop_duplicates(subset='id')

    df_test_features_copy['adjusted_time'] = df_test_features_copy.groupby('id')['time_milisecond'].transform(lambda x: x - x.min())
    df_test_features_copy['max_adjusted_time'] = df_test_features_copy.groupby('id')['adjusted_time'].transform('max')
    df_test_features_copy['new_start_time'] = df_test_features_copy.groupby('id')['time_milisecond'].transform('min')

    # Drop duplicates to keep only one row per id with the calculated values
    df_test_features_unique = df_test_features_copy[['id', 'max_adjusted_time', 'new_start_time']].drop_duplicates()

    # Merge df_predicted with df_test_features_unique to get the necessary columns
    df_predicted = df_predicted.merge(df_test_features_unique, on='id', how='left')

    # Revert the normalized time back to seconds
    df_predicted[f'{col_name}_in_seconds'] = (df_predicted[col_name] * df_predicted['max_adjusted_time'] + df_predicted['new_start_time']) / 1000.0

    # Drop the temporary columns used for the calculation
    df_predicted.drop(columns=['max_adjusted_time', 'new_start_time'], inplace=True)

    # Drop the original 'start_move' column
    df_predicted.drop(columns=[col_name], inplace=True)

    return df_predicted


def plot_trajectory_to_file(df_predicted, df_test_features, filename):
    # Randomly select 10 unique ids from the test data
    random_indices = random.sample(sorted(df_test_features['id'].unique()), 10)

    # Create subplots
    fig, axes = plt.subplots(5, 2, figsize=(8, 20))  
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        ax = axes[i]
        # Extract the DataFrame for the current trajectory
        df = df_test_features[df_test_features['id'] == idx]
        
        # Plot the trajectory
        ax.plot(df['scaled_rotation_x'], df['scaled_rotation_y'], marker='o', alpha=0.5, markersize=3)
        
        # Use the 'id' to get the predicted start move time
        predicted_time_start = df_predicted.loc[df_predicted['id'] == idx, 'start_move_in_seconds'].values[0]
        predicted_idx_start = ((df['time_milisecond']/1000) - predicted_time_start).abs().idxmin()
        predicted_x_start = df.loc[predicted_idx_start, 'scaled_rotation_x']
        predicted_y_start = df.loc[predicted_idx_start, 'scaled_rotation_y']
        ax.plot(predicted_x_start, predicted_y_start, marker='o', color='orange', label='Start Movement', markersize=8)
        
        # Check if there is a stop_turn time for this trajectory
        if 'stop_turn_in_seconds' in df_predicted.columns and not df_predicted.loc[df_predicted['id'] == idx, 'stop_turn_in_seconds'].isna().all():
            predicted_time_turn = df_predicted.loc[df_predicted['id'] == idx, 'stop_turn_in_seconds'].values[0]
            predicted_idx_turn = ((df['time_milisecond']/1000) - predicted_time_turn).abs().idxmin()
            predicted_x_turn = df.loc[predicted_idx_turn, 'scaled_rotation_x']
            predicted_y_turn = df.loc[predicted_idx_turn, 'scaled_rotation_y']
            ax.plot(predicted_x_turn, predicted_y_turn, marker='o', color='red', label='Turning Point', markersize=8)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.set_title(f'Trajectory {idx}')
    
    fig.suptitle('Predictions', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.savefig(filename, dpi=300)

def predict_and_plot_start(df_test_features, model_name, features, model_path):
    col_name = model_name.split("_model")[0] 
    df_predicted = make_predictions(df_test_features, col_name, features, model_path)
    edited_df_predicted = edit_predictions(df_predicted, df_test_features, 'start_move')
    return edited_df_predicted

def predict_and_plot_turn(df_test_features, model_name, features, model_path):
    col_name = model_name.split("_model")[0]
    df_test_features_turn = df_test_features[(~df_test_features['id_update'].isna()) & (df_test_features['id_target'] != df_test_features['id_update'])]
    df_predicted = make_predictions(df_test_features_turn, col_name, features, model_path)
    edited_df_predicted = edit_predictions(df_predicted, df_test_features_turn, 'stop_turn')
    return edited_df_predicted
