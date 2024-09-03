import scipy.io
import os
import pandas as pd
import numpy as np

#example use for file: convert_raw_to_df('/Users/avitalvasiliev/Documents/GitHub/ExtractingTurningPointGUI/t011222')

def extract_data_from_info(day_dir):  # info of entire day
    Info_dir = os.path.join(day_dir, 'Info')
    Info_file = [f for f in os.listdir(Info_dir) if f.endswith('param.mat')][0]
    Info_file_path = os.path.join(Info_dir, Info_file)
    Info_data = scipy.io.loadmat(Info_file_path)
    recording_day_id = Info_data['DDFparam']['ID'].flatten()[0]
    SESSparam = Info_data['SESSparam']
    type_list = SESSparam['fileConfig'].flatten()[0]['HFS'].flatten().astype(int)
    session_list = SESSparam['SubSess'].flatten()[0]['Files'].flatten()[0].tolist()
    return recording_day_id, type_list, session_list

def extract_data_from_edfiles_ed2videomap(day_dir, recording_day_id, type_list, session_list):
    ED2videomap_dir = os.path.join(day_dir, 'ED2videomap')
    EDfiles_dir = os.path.join(day_dir, 'EDfiles')
    temp = []  # Empty list for dataframes
    video_ticks_total = {}  # To hold filtered video_ticks data

    # Loop through all MAT files in the folder
    for i, file_name in enumerate(os.listdir(EDfiles_dir)):
        ed_file_path = os.path.join(EDfiles_dir, file_name)
        map_file_path = os.path.join(ED2videomap_dir, file_name)
        file_index = os.path.splitext(file_name)[0].split('.')[-1]

        # Load the ED file
        ed_data = scipy.io.loadmat(ed_file_path, struct_as_record=True, squeeze_me=True)
        failed_non_failed = ed_data['trials'][:, 2]
        trial_events = ed_data['TrialTimes']
        id_target = ed_data['bhvStat'][:, 2]
        id_update = ed_data['bhvStat'][:, 3]
        video_ticks = ed_data['VideoTicks']

        # Load the map file
        ED2videomap = scipy.io.loadmat(map_file_path, struct_as_record=True, squeeze_me=True)['ed2video']

        # Extract the relevant columns
        df = pd.DataFrame(np.round(ED2videomap), columns=['trial_number_in_file', 'id'])
        df['failed_non_failed'] = failed_non_failed
        df_non_failed = df[df['failed_non_failed'] == 1].copy()
        df_non_failed['file_number'] = int(file_index)
        df_non_failed['recording_day'] = int(recording_day_id)
        df_non_failed['type'] = df_non_failed['file_number'].apply(lambda idx: 'HFS' if type_list[int(idx) - 1] == 1 else 'Control')
        for session_number, (start, end) in enumerate(session_list, start=1):
            if start <= int(file_index) <= end:
                df_non_failed['session_number'] = session_number
                break

        df_non_failed['start_time'] = trial_events[:, 0]
        df_non_failed['end_time'] = trial_events[:, 10]
        df_non_failed['cue_time'] = trial_events[:, 3]
        df_non_failed['go_time'] = trial_events[:, 4]
        df_non_failed['update_time'] = trial_events[:, 5]
        df_non_failed['move_time'] = trial_events[:, 6]
        df_non_failed['id_target'] = id_target
        df_non_failed['id_update'] = id_update

        # Filter video_ticks to match valid trials
        valid_trials = (df_non_failed['trial_number_in_file'].values - 1).astype(int)  # Adjust for 0-based index
        video_ticks_filtered = {}

        for trial in valid_trials:
            timestamps = video_ticks[trial][0][0]
            timestamp_index = video_ticks[trial][0][1]
            filtered_timestamps = timestamps[timestamp_index != 0]
            video_ticks_filtered[trial+1] = filtered_timestamps

        video_ticks_total[int(file_index)] = video_ticks_filtered

        temp.append(df_non_failed)

    # Concatenate all DataFrames into one    
    final_df = pd.concat(temp, ignore_index=True)
    # Sort the DataFrame by 'file_number' and then by 'trial_number'
    final_df_sorted = final_df.sort_values(by=['file_number', 'trial_number_in_file'], ascending=[True, True])
    final_df_sorted = final_df_sorted.reset_index(drop=True)
    final_df_sorted.dropna(subset=['id'], inplace=True)
    final_df_sorted['id'] = final_df_sorted['id'].astype(int)
    final_df_sorted['trial_number_in_file'] = final_df_sorted['trial_number_in_file'].astype(int)
    final_df_sorted['file_number'] = final_df_sorted['file_number'].astype(int)
    final_df_sorted['id_target'] = final_df_sorted['id_target'].astype(int)

    return final_df_sorted, video_ticks_total

def extract_data_from_csv(day_dir, final_df_sorted):
    dfs = []
    csv_dir = os.path.join(day_dir, 'DLC_color')
    indices = final_df_sorted['id'].astype(int)
    for index in indices:
        pattern = "trial" + str(index) + "-"
        for i, file_name in enumerate(os.listdir(csv_dir)):
            if pattern in file_name:
                csv_file_path = os.path.join(csv_dir, file_name)
                headers = pd.read_csv(csv_file_path, nrows=3)
                combined_header = ['_'.join(col) for col in zip(headers.iloc[0], headers.iloc[1])]
                df = pd.read_csv(csv_file_path, skiprows=3, names=combined_header)
                # Calculate the median for the relevant 'x' and 'y' columns
                x_columns = ['tip2_x', 'tip3_x', 'tip4_x', 'tip5_x']
                x_median = np.median(df[x_columns], axis=1)
                y_columns = ['tip2_y', 'tip3_y', 'tip4_y', 'tip5_y']
                y_median = np.median(df[y_columns], axis=1)
                # Create a new DataFrame with 'id', 'x_median', and 'y_median'
                temp_df = pd.DataFrame({'id': index, 'x': x_median, 'y': y_median})
                # Append the DataFrame to the list
                dfs.append(temp_df)
    coord_df = pd.concat(dfs, ignore_index=True)
    return coord_df

def add_timestamps(merged_df, video_ticks_total):
    for file_number, trial_data in video_ticks_total.items():
        for trial_number, timestamps in trial_data.items():
            # Extract the subset of the DataFrame
            curr = merged_df[(merged_df['file_number'] == file_number) & (merged_df['trial_number_in_file'] == trial_number)]
            video_array = video_ticks_total[file_number][trial_number]

            df_length = len(curr)
            array_length = len(video_array)

            # Pad the array with NaN values if it's smaller than the DataFrame
            if array_length < df_length:
                padded_array = np.pad(video_array, (0, df_length - array_length), constant_values=np.nan)
            elif array_length > df_length:
                padded_array = video_array[:df_length]
            else:
                padded_array = video_array

            # Convert the padded array into a Pandas Series
            time_series = pd.Series(padded_array)

            # Ensure the index of time_series matches the index of curr
            time_series.index = curr.index

            # Assign the Series to the 'time' column of the DataFrame using .loc
            merged_df.loc[curr.index, 'time_milisecond'] = time_series

def convert_raw_to_df(day_dir):
    recording_day_id, type_list, session_list = extract_data_from_info(day_dir)
    events_df_sorted, video_ticks_total = extract_data_from_edfiles_ed2videomap(day_dir, recording_day_id, type_list, session_list)
    coord_df = extract_data_from_csv(day_dir, events_df_sorted)
    merged_df = events_df_sorted.merge(coord_df, on='id', how='left')
    merged_df.drop(columns=['failed_non_failed'], inplace=True)
    add_timestamps(merged_df, video_ticks_total)
    return merged_df

