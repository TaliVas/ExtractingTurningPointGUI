import scipy.io
import os
import pandas as pd
import numpy as np
import glob
import re 

def extract_csv(day_dir):
    Info_dir = os.path.join(day_dir, 'Info')
    EDfiles_folder_dir = os.path.join(day_dir, 'EDfiles')

    # Read CSV files
    file_path = glob.glob(os.path.join(Info_dir, '*ED2VideoTrialMap.csv'))[0]
    ED2videomap = pd.read_csv(file_path, index_col=None).astype({'ED_trial': 'int', 'Video_trial': 'Int64'})

    SESSparam = pd.read_csv(os.path.join(Info_dir, 'SESSparam.csv'), index_col=None)
    type_list = SESSparam['HFS'].astype(int)
    subsession_files_list = SESSparam['Files'].unique()
    session_list = [list(map(int, item.split('-'))) for item in subsession_files_list]

    DDFparam = pd.read_csv(os.path.join(Info_dir, 'DDFparam.csv'), index_col=None)

    EDfiles = pd.read_csv(os.path.join(EDfiles_folder_dir, 'EDfiles.csv'), usecols=range(22), index_col=False)
    VideoTicks = pd.read_csv(os.path.join(EDfiles_folder_dir, 'VideoTicks.csv'), header=None, low_memory=False, index_col=None)

    # Sorting function to correctly order filenames
    def natural_sort_key(file_name):
        """Extract recording date, session number, and file number for sorting."""
        match = re.match(r'([a-zA-Z]*\d+)(\d+)ee\.(\d+)', file_name)
        if match:
            recording_day = int(re.search(r'\d+', match.group(1)).group())  # Extract recording date (e.g., '56' from 't56')
            session_number = int(match.group(2))  # Extract session number (e.g., '01' from '01ee')
            file_number = int(match.group(3))  # Extract file number (e.g., '1' from '.1')
            return (recording_day, session_number, file_number)
        return (float('inf'), float('inf'), float('inf'))  # Default to avoid errors

    # Sort the file names correctly
    EDfiles = EDfiles.sort_values(by='EDfile_name', key=lambda x: x.map(natural_sort_key))
    # Initialize DataFrame
    df = pd.DataFrame(ED2videomap[['ED_trial', 'Video_trial']])
    df = df.rename(columns={'ED_trial': 'trial_number_in_file', 'Video_trial': 'id'})
    df['failed_non_failed'] = EDfiles['failed/non_failed'].values
    df['file_name'] = EDfiles['EDfile_name'].values  # Use sorted order
    df['file_name'] = pd.Categorical(df['file_name'], categories=EDfiles['EDfile_name'].unique(), ordered=True)
    df['file_number'] = df['file_name'].astype('category').cat.codes + 1


    df['recording_day'] = DDFparam['ID'][0]
    df['type'] = df['file_number'].apply(lambda idx: 'HFS' if type_list[int(idx) - 1] == 1 else 'Control')

    # Assign session numbers
    df['session_number'] = None
    for index, row in df.iterrows():
        file_index = row['file_number']
        for session_number, (start, *end) in enumerate(session_list, start=1):
            end = end[0] if end else start
            if start <= file_index <= end:
                df.at[index, 'session_number'] = session_number
                break

    # Add additional time fields
    df['start_time'] = EDfiles['trial_mode'].values
    df['end_time'] = EDfiles['end_of_trial'].values
    df['cue_time'] = EDfiles['cue_onset'].values
    df['go_time'] = EDfiles['go_signal'].values
    df['update_time'] = EDfiles['trial_jump'].values
    df['move_time'] = EDfiles['move_onset'].values
    df['id_target'] = EDfiles['id_target'].values
    df['id_update'] = EDfiles['id_update'].values

    # Process VideoTicks file
    file_names = VideoTicks.iloc[0, 1:].str.split('.').str[0:3].str.join('.')
    trial_indices = VideoTicks.iloc[1, 1:]
    timestamps = VideoTicks.iloc[2:, 1:].reset_index(drop=True).astype(float)

    video_ticks_df = pd.DataFrame({'file_name': file_names, 'trial_number_in_file': trial_indices}).join(timestamps.T)
    video_ticks_df = video_ticks_df.dropna(subset=['file_name'])
    video_ticks_df['trial_number_in_file'] = video_ticks_df['trial_number_in_file'].astype(int)

    video_ticks_long = (video_ticks_df.set_index(['file_name', 'trial_number_in_file']).iloc[:, :].stack().reset_index()
        .rename(columns={0: 'time_milisecond'}))

    # Merge and clean final DataFrame
    merged = df.merge(video_ticks_long, how='inner', on=['file_name', 'trial_number_in_file'])
    merged = merged.dropna(subset=['id'])

    df_non_failed = merged[merged['failed_non_failed'] == 1].copy()
    df_non_failed.dropna(subset=['time_milisecond'], inplace=True)
    df_non_failed = df_non_failed.sort_values(by=['file_number', 'trial_number_in_file'], ascending=[True, True])
    df_non_failed = df_non_failed.drop(columns='level_2', errors='ignore')  # Avoid error if column doesn't exist
    return df_non_failed

def extract_data_from_csv(day_dir, df):
    dfs = []
    csv_dir = os.path.join(day_dir, 'DLC_color')
    indices = df['id'].unique().astype(int)
    for index in indices:
        pattern = "trial" + str(index) + "-"
        for i, file_name in enumerate(os.listdir(csv_dir)):
            if pattern in file_name:
                if not file_name.endswith('filtered.csv'):
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

def trim_and_merge_df(df1, df2):
    df1_len = len(df1)
    df2_len = len(df2)
    min_len = min(df1_len, df2_len)
    df1_trimmed = df1.iloc[:min_len]
    df2_trimmed = df2.iloc[:min_len]
    merged_df = pd.concat([df1_trimmed.reset_index(drop=True), df2_trimmed.reset_index(drop=True)], axis=1)
    return merged_df

def merge_total_dfs(coord_df, df_non_failed):
    merged_dfs = []
    
    unique_ids = coord_df['id'].unique().astype(int)

    for unique_id in unique_ids:
        df1_filtered = coord_df[coord_df['id'] == unique_id]
        df2_filtered = df_non_failed[df_non_failed['id'] == unique_id]

        merged_df = trim_and_merge_df(df1_filtered, df2_filtered)
        merged_dfs.append(merged_df)  # Append merged DataFrame

    if merged_dfs:
        final_merged_df = pd.concat(merged_dfs, ignore_index=True)
    else:
        final_merged_df = pd.DataFrame()  # Return an empty DataFrame if nothing is merged

    return final_merged_df



def convert_raw_2_df(day_dir):
    df_non_failed = extract_csv(day_dir)
    coord_df = extract_data_from_csv(day_dir, df_non_failed)
    final_merged_df = merge_total_dfs(coord_df, df_non_failed)
    final_merged_df = final_merged_df.reset_index()
    final_merged_df = final_merged_df.loc[:, ~final_merged_df.columns.duplicated()]
    return final_merged_df