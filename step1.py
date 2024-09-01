import scipy.io
import os
import pandas as pd
import numpy as np

def extract_data_from_info(day_dir): #info of entire day
    Info_dir = os.path.join(day_dir, 'Info')
    Info_file = [f for f in os.listdir(Info_dir) if f.endswith('param.mat')][0]
    Info_file_path = os.path.join(Info_dir , Info_file)
    Info_data = scipy.io.loadmat(Info_file_path)
    recording_day_id = Info_data['DDFparam']['ID'][0][0][0][0]
    SESSparam = Info_data['SESSparam']
    fileConfig = SESSparam['fileConfig'][0, 0]
    HFS_field = fileConfig['HFS'][0]
    type_list = np.array([int(x[0][0]) for x in HFS_field])
    SubSess = SESSparam['SubSess'][0, 0]
    Files_field = SubSess['Files'][0][0]
    session_list = Files_field.tolist()

    return recording_day_id, type_list, session_list

def extract_data_from_edfiles_ed2videomap(day_dir, recording_day_id, type_list, session_list):
    ED2videomap_dir = os.path.join(day_dir, 'ED2videomap')
    EDfiles_dir = os.path.join(day_dir, 'EDfiles')
    temp = [] # Empty list for dataframes

    # Loop through all MAT files in the folder
    for i, file_name in enumerate(os.listdir(EDfiles_dir)):
        ed_file_path = os.path.join(EDfiles_dir, file_name)
        map_file_path = os.path.join(ED2videomap_dir, file_name)
        file_index = os.path.splitext(file_name)[0].split('.')[-1]
        
        # Load the ED file
        ed_data = scipy.io.loadmat(ed_file_path)
        failed_non_failed = ed_data['trials'][:, 2]
        trial_events = ed_data['TrialTimes']
        id_target = ed_data['bhvStat'][:, 2]
        id_update = ed_data['bhvStat'][:, 3]
        video_ticks = ed_data['videoTicks']
        # Load the map file
        ED2videomap = scipy.io.loadmat(map_file_path)['ed2video']

        # Extract the relevant columns
        df = pd.DataFrame(np.round(ED2videomap), columns=['trial_number_in_file', 'DLC_color_id'])
        df['failed_non_failed'] = failed_non_failed
        df_non_failed = df[df['failed_non_failed'] == 1].copy()
        df_non_failed['file_number'] = int(file_index)
        df_non_failed['recording_day'] = recording_day_id
        df_non_failed['type'] = df_non_failed['file_number'].apply(lambda idx: 'HFS' if type_list[int(idx)-1] == 1 else 'Control')
        for session_number, (start, end) in enumerate(session_list, start=1):
            if start <= int(file_index) <= end:
                df_non_failed['session_number'] = session_number
                break

        df_non_failed['start_time'] = trial_events[:, 0]
        df_non_failed['end_time'] = trial_events[:, 10]
        df_non_failed['que_time'] = trial_events[:, 3]
        df_non_failed['go_time'] = trial_events[:, 4]
        df_non_failed['update_time'] = trial_events[:, 5]
        df_non_failed['move_time'] = trial_events[:, 6]
        df_non_failed['id_target'] = id_target
        df_non_failed['id_update'] = id_update
        temp.append(df_non_failed)
    # Concatenate all DataFrames into one    
    final_df = pd.concat(temp, ignore_index=True)
    # Sort the DataFrame by 'file_number' and then by 'trial_number'
    final_df_sorted = final_df.sort_values(by=['file_number','trial_number_in_file'], ascending=[True,True])
    final_df_sorted = final_df_sorted.reset_index(drop=True)
    final_df_sorted.dropna(subset=['DLC_color_id'], inplace=True)
    final_df_sorted['DLC_color_id'] = final_df_sorted['DLC_color_id'].astype(int)
    final_df_sorted['trial_number_in_file'] = final_df_sorted['trial_number_in_file'].astype(int)
    final_df_sorted['file_number'] = final_df_sorted['file_number'].astype(int)
    final_df_sorted['id_target'] = final_df_sorted['id_target'].astype(int)

    return final_df_sorted

def extract_data_from_csv(day_dir, final_df_sorted):
    dfs = []
    csv_dir = os.path.join(day_dir, 'DLC_color')
    indices = final_df_sorted['DLC_color_id'].astype(int)
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
                temp_df = pd.DataFrame({'DLC_color_id': index, 'x': x_median,'y': y_median})
                # Append the DataFrame to the list
                dfs.append(temp_df)
    coord_df = pd.concat(dfs, ignore_index=True)
    return coord_df

def merge_df(day_dir):
    recording_day_id, type_list, session_list = extract_data_from_info(day_dir)
    events_df_sorted = extract_data_from_edfiles_ed2videomap(day_dir, recording_day_id, type_list, session_list)
    coord_df = extract_data_from_csv(day_dir, events_df_sorted)
    merged_df = events_df_sorted.merge(coord_df, on='DLC_color_id', how='left')
    merged_df.drop(columns=['failed_non_failed'], inplace=True)
    return merged_df

merged_df = merge_df(day_dir = '/Users/avitalvasiliev/Documents/GitHub/ExtractingTurningPointGUI/t011222')