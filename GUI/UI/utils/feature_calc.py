from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

def data_preprocessing(df, min_len):
    try:
        df = df.dropna(subset=['time_milisecond', 'id'])
        df = df.groupby('id').filter(lambda x: len(x) >= min_len)
        return df

    except Exception as e:
        print(f"Error in data_preprocessing: {e}")
        raise


def process_dataframe(len_df):
    """
    Create processed df.
    """
    process_df = []
    for trial_id in len_df['id'].unique():
        trial = process_trial(len_df, trial_id)
        process_df.append(trial)
    process_df = pd.concat(process_df, ignore_index=True)
    return process_df

def process_trial(df, trial_id):
    """
    Process a single trial by updating columns, computing velocity, and updating movement times.
    """
    trajectory = df[df['id'] == trial_id].copy() #all data points with the same id
    initial_start_time = trajectory.iloc[0]['start_time']
    
    # Adjust time columns
    time_columns = ['start_time', 'end_time', 'cue_time', 'go_time', 'update_time', 'move_time','time_milisecond']
    for col in time_columns:
        trajectory[col] -= initial_start_time
    trajectory['time_milisecond'] *= 1000
    trajectory['cue_milisecond'] = trajectory.iloc[0]['cue_time'] * 1000
    trajectory['go_milisecond'] = trajectory.iloc[0]['go_time'] * 1000
    trajectory = compute_velocities(trajectory, trajectory.iloc[0]['cue_time'] * 1000)
    return trajectory


def compute_velocities(df, signal_cue):
    """
    Calculate velocity for movements post-signal cue.
    """
    # Filter and calculate differences
    df_filtered = df[df['time_milisecond'] >= signal_cue].copy()
    dx = df_filtered['x'].diff()
    dy = df_filtered['y'].diff()
    dt = df_filtered['time_milisecond'].diff()
    
    # Compute velocity while avoiding division by zero
    df_filtered['velocity'] = np.sqrt(dx**2 + dy**2) / (dt + 1e-10)
    ddx = dx.diff().fillna(0)
    ddy = dy.diff().fillna(0)
    df_filtered['curvature'] = (dx * ddy - dy * ddx) / (np.power(dx ** 2 + dy ** 2, 1.5) + 1e-10)

    return df_filtered.iloc[1:]  # skip the first NaN result from diff()(0)

def window_and_normalize_trial(trial, right_window=1500, left_window=200):
    """
    Process and window a single trial.
    """
    signal_go = trial.iloc[0]['go_milisecond']

    # Determine the window around the go signal
    window_start = signal_go - left_window
    window_end = signal_go + right_window

    # Filter the trial to the specified window
    windowed_trial = trial[(trial['time_milisecond'] >= window_start) & (trial['time_milisecond'] <= window_end)].copy()
    new_start_time = windowed_trial.iloc[0]['time_milisecond']

    # Adjust times to start from zero within the window
    windowed_trial['adjusted_time'] = windowed_trial['time_milisecond'] - new_start_time

    # Normalize the adjusted times
    max_adjusted_time = windowed_trial['adjusted_time'].max()
    windowed_trial['normalized_time'] = windowed_trial['adjusted_time'] / max_adjusted_time

    return windowed_trial

def window_trials(processed_df, right_window=1500, left_window=200):
    """
    Adjust times for each trial in the processed DataFrame.
    """
    # Apply the function to all trials in the dataframe
    windowed_trials = []
    for trial_id in processed_df['id'].unique():
        trial = processed_df[processed_df['id'] == trial_id]
        windowed_trial = window_and_normalize_trial(trial, right_window, left_window)
        windowed_trials.append(windowed_trial)

    return pd.concat(windowed_trials, ignore_index=True)

def apply_smoothing(group, col_x='x', col_y='y', window_length=5, poly_order=1):
    """
    Applies Savitzky-Golay smoothing to specified columns in a DataFrame group.
    """
    # Apply the Savitzky-Golay filter
    x_smooth = savgol_filter(group[col_x], window_length, poly_order)
    y_smooth = savgol_filter(group[col_y], window_length, poly_order)

    # Add smoothed data to new columns
    group['smooth_x'] = x_smooth
    group['smooth_y'] = y_smooth

    return group

def recenter_trajectory(group):
    """
    Recenters a trajectory group by subtracting the coordinates at the start movement time.
    """

    initial_x_offset = group['smooth_x'].iloc[0]
    initial_y_offset = group['smooth_y'].iloc[0]

    # Apply the offset to recenter the trajectory
    group['centered_x'] = group['smooth_x'] - initial_x_offset
    group['centered_y'] = group['smooth_y'] - initial_y_offset

    return group

def apply_rotation(group):
    """
    Applies rotation and reflection transformations to trajectory data.
    """
    # Extract target ID from the first row of the group
    id_target = group.iloc[0]['id_target']
    # Initialize target angles for rotation
    target_angles = {1: 270, 2: 315, 3: 0, 4: 45, 5: 90, 6: 135, 7: 180, 8: 225}

    # Check for updates and obtain update target ID if present
    is_updated = not pd.isna(group.iloc[0]['id_update'])
    id_target_update = group.iloc[0]['id_update'] if is_updated else None

    if id_target == 5:
        if is_updated and (id_target_update > id_target):
            # Reflect over the Y-axis if movement is not clockwise
            group['rotation_x'],group['rotation_y']  = -group['centered_x'], group['centered_y']
        else:
            # No rotation needed
            group['rotation_x'], group['rotation_y'] = group['centered_x'], group['centered_y']
    else:
        # General case for all other targets
        rotation_angle = 90 - target_angles[id_target] 
        # Rotate coordinates based on computed angle
        group['rotation_x'], group['rotation_y'] = rotate_trajectory(group['centered_x'], group['centered_y'], rotation_angle)
        if is_updated and (id_target_update > id_target):
            # Reflect over the Y-axis if movement is not clockwise
            group['rotation_x'], group['rotation_y'] =  -group['centered_x'], group['centered_y']
    return group

def rotate_trajectory(x, y, angle):
    """
    Rotates a point around the origin (0, 0) by a specified angle.
    """
    angle_rad = np.radians(angle)
    x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_new, y_new
def map_targets_to_angle(id_target, id_update):
    """
    Maps target IDs to angles.
    """
    if not pd.isna(id_update):
        old_target, new_target = round(id_target), round(id_update)
        if not np.isnan(new_target):
            target_diff = abs(new_target - old_target)
            # Adjust for circular nature of targets.
            if target_diff > 4:
                target_diff = 8 - target_diff
            angle_map = {1: 45, 2: 90, 3: 135, 4: 180}
            return angle_map.get(target_diff, 0)
    else:
        return 0
    
def calc_points(group):
    """
    Calculate the angles between consecutive triplets of points in the group.
    """
    group['point'] = list(zip(group['rotation_x'], group['rotation_y']))
    angles = []
    for i in range(len(group) - 2):
        A = group['point'].iloc[i]
        B = group['point'].iloc[i + 1]
        C = group['point'].iloc[i + 2]
        angle = angle_between_three_points(A, B, C)
        angles.append(angle)
    # Add NaN for the last two points as they don't have enough subsequent points for the angle calculation
    angles.extend([np.nan, np.nan])
    group['angle'] = angles
    return group

def angle_between_three_points(A, B, C):
    # Convert points to numpy arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    # Calculate vectors AB and BC
    AB = B - A
    BC = C - B
    # Calculate the dot product and magnitudes of AB and BC
    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_AB * magnitude_BC)
    # Handle potential numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # Calculate the angle in radians
    angle_radians = np.arccos(cos_angle)
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def calculate_relative_angle_to_end(group):
    end_x, end_y = group['rotation_x'].iloc[-1], group['rotation_y'].iloc[-1]
    delta_x_to_end = end_x - group['rotation_x']
    delta_y_to_end = end_y - group['rotation_y']
    
    start_x, start_y = group['rotation_x'].iloc[0], group['rotation_y'].iloc[0]
    delta_x_start_to_end = end_x - start_x
    delta_y_start_to_end = end_y - start_y
    
    angle_to_end = np.arctan2(delta_y_to_end, delta_x_to_end)
    angle_start_to_end = np.arctan2(delta_y_start_to_end, delta_x_start_to_end)
    relative_angles = np.degrees(angle_to_end - angle_start_to_end)
    relative_angles = (relative_angles + 180) % 360 - 180
    group['angle_to_end'] = relative_angles
    group['distance_to_end'] = np.sqrt((group['rotation_x'] - group['rotation_x'].iloc[-1])**2 + (group['rotation_y'] - group['rotation_y'].iloc[-1])**2)
    return group

def scale_features(df, features, group_col):
    """
    Scale specified features within each group defined by group_col.
    """
    df_final = df.copy()
    scaler = StandardScaler()
    for feature in features:
        scaled_feature = 'scaled_' + feature
        df_final[scaled_feature] = df_final.groupby(group_col)[feature].transform(lambda group: scaler.fit_transform(group.values.reshape(-1, 1)).flatten())
    return df_final

def feature_calc(df):
    try:
        try:
            len_df = data_preprocessing(df, min_len=10)
        except Exception as e:
            print(f"Error in data_preprocessing: {e}")
            return None  # Return None if it fails

        try:
            processed_df = process_dataframe(len_df)
        except Exception as e:
            print(f"Error in process_dataframe: {e}")
            return None  # Return None if it fails

        try:
            df_windowed = window_trials(processed_df)
        except Exception as e:
            print(f"Error in window_trials: {e}")
            return None

        try:
            df_smooth = df_windowed.groupby('id').apply(apply_smoothing).reset_index(drop=True)
        except Exception as e:
            print(f"Error in apply_smoothing: {e}")
            return None

        try:
            df_centered = df_smooth.groupby('id').apply(recenter_trajectory).reset_index(drop=True)
        except Exception as e:
            print(f"Error in recenter_trajectory: {e}")
            return None

        try:
            df_rotation = df_centered.groupby('id').apply(apply_rotation).reset_index(drop=True)
        except Exception as e:
            print(f"Error in apply_rotation: {e}")
            return None

        try:
            df_rotation['type_trajectory'] = df_rotation.apply(lambda row: map_targets_to_angle(row['id_target'], row['id_update']), axis=1)
        except Exception as e:
            print(f"Error in map_targets_to_angle: {e}")
            return None

        try:
            df_features_angle = df_rotation.groupby('id').apply(calc_points).reset_index(drop=True)
        except Exception as e:
            print(f"Error in calc_points: {e}")
            return None

        try:
            df_features_angle_to_end = df_features_angle.groupby('id').apply(calculate_relative_angle_to_end).reset_index(drop=True)
        except Exception as e:
            print(f"Error in calculate_relative_angle_to_end: {e}")
            return None

        try:
            features_to_scale = ['rotation_x', 'rotation_y', 'velocity', 'angle', 'angle_to_end', 'curvature', 'distance_to_end']
            df_final = scale_features(df_features_angle_to_end, features_to_scale, 'type_trajectory')
        except Exception as e:
            print(f"Error in scale_features: {e}")
            return None

        try:
            df_features_final = df_final.dropna(axis=0, subset=features_to_scale)
        except Exception as e:
            print(f"Error in dropna: {e}")
            return None

        return df_features_final  # Return final processed DataFrame

    except Exception as e:
        print(f"Unexpected error in feature_calc: {e}")
        return None  # Ensure function exits safely on unexpected errors
