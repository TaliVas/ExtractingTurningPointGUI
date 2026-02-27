% Folder containing the ED files
folderPath = 'EDfiles'; % Change this to the actual path
outputFile = fullfile(folderPath, 'EDfiles.csv'); % Output CSV in the same folder

% Get all .mat files in the folder
fileList = dir(fullfile(folderPath, '*.mat'));

% Extract components from filenames
fileNames = {fileList.name};

% Initialize arrays to store parsed values
sessionNumbers = zeros(size(fileNames));
fileNumbers = zeros(size(fileNames));

for i = 1:length(fileNames)
    % Extract session number and file number, ignoring the recording date
    tokens = regexp(fileNames{i}, '[a-zA-Z]*\d+(\d+)ee\.(\d+)\.mat', 'tokens');
    
    if isempty(tokens)
        error('Filename format not recognized: %s', fileNames{i});
    end
    
    tokens = tokens{1}; % Extract matched tokens
    
    sessionNumbers(i) = str2double(tokens{1}); % Extract session number
    fileNumbers(i) = str2double(tokens{2}); % Extract file number
end

% Sort based on session number -> file number
[~, sortIdx] = sortrows([sessionNumbers', fileNumbers']);

% Reorder fileList based on sorted indices
fileList = fileList(sortIdx);

% Initialize storage for combined data
combinedData = [];

% Column headers
columnHeaders = [{'EDfile_name'}, ...
                 {'index_events_code_start', 'index_events_code_end', 'failed/non_failed', 'update/non-update', 'id_target', 'id_update'}, ...
                 {'bhvStat1', 'failed/non_failed', 'id_target', 'id_update'}, ...
                 {'trial_mode', 'start_of_pre_cue', 'last_enter_to_target', ...
                  'cue_onset', 'go_signal', 'trial_jump', ...
                  'move_onset', 'in_periphery', 'end_of_hold', 'reward', 'end_of_trial'}];

% Process each file
for iFile = 1:length(fileList)
    % Load the current file
    fileName = fileList(iFile).name;
    filePath = fullfile(folderPath, fileName);
    loadedData = load(filePath);

    % Extract data from 'trials', 'bhvStat', and 'TrialTimes'
    trials = loadedData.trials; % 6 columns
    bhvStat = loadedData.bhvStat; % 4 columns
    TrialTimes = loadedData.TrialTimes; % 11 columns

    % Initialize successful trial index
    successIndex = 1;

    % Loop through each trial
    for iTrial = 1:size(trials, 1)
        % Initialize row with file name
        rowData = {fileName};

        % Add 'trials' data (exact structure)
        rowData = [rowData, num2cell(trials(iTrial, :))];

        % Check success/failure (3rd column of 'trials')
        if trials(iTrial, 3) == 1 % Success
            % Add 'bhvStat' and 'TrialTimes' data if indices match
            if successIndex <= size(bhvStat, 1) && successIndex <= size(TrialTimes, 1)
                rowData = [rowData, num2cell(bhvStat(successIndex, :)), num2cell(TrialTimes(successIndex, :))];
            else
                error('Mismatch in successful trial indices for bhvStat and TrialTimes.');
            end
            successIndex = successIndex + 1; % Move to the next success
        else % Failure
            % Add NaNs for 'bhvStat' and 'TrialTimes'
            rowData = [rowData, num2cell(NaN(1, size(bhvStat, 2))), num2cell(NaN(1, size(TrialTimes, 2)))];
        end

        % Append the row to combined data
        combinedData = [combinedData; rowData];
    end
end

% Open the file for writing
fid = fopen(outputFile, 'w');
if fid == -1
    error('Could not open file %s for writing. Check folder path or permissions.', outputFile);
end

% Write column headers
fprintf(fid, '%s,', columnHeaders{1:end-1});
fprintf(fid, '%s\n', columnHeaders{end});

% Write data rows
for iRow = 1:size(combinedData, 1)
    rowData = combinedData(iRow, :);

    % Convert numeric arrays and cell values properly
    for j = 1:length(rowData)
        if isnumeric(rowData{j}) && ~isscalar(rowData{j})
            rowData{j} = sprintf('[%s]', strjoin(arrayfun(@num2str, rowData{j}, 'UniformOutput', false), ' '));
        elseif isnumeric(rowData{j}) && isscalar(rowData{j})
            rowData{j} = num2str(rowData{j});
        elseif ischar(rowData{j})
            rowData{j} = rowData{j};
        else
            rowData{j} = 'NA'; % Default for unsupported data types
        end
    end

    % Write row to CSV with proper formatting
    fprintf(fid, '%s,', rowData{1:end-1});
    fprintf(fid, '%s\n', rowData{end});
end

% Close the file
fclose(fid);

disp(['Combined data exported to ', outputFile]);
