% Folder containing the ED files
folderPath = 'EDfiles'; % Change to the correct folder path
outputFile = fullfile(folderPath, 'VideoTicks.csv'); % Output CSV file

% Get all .mat files in the folder
fileList = dir(fullfile(folderPath, '*.mat'));

% Extract components from filenames
fileNames = {fileList.name};

% Initialize arrays to store parsed values
sessionNumbers = zeros(size(fileNames));
fileNumbers = zeros(size(fileNames));

for i = 1:length(fileNames)
    % Extract session number and file number (ignore recording date)
    tokens = regexp(fileNames{i}, '[a-zA-Z]*\d+(\d+)ee\.(\d+)\.mat', 'tokens');
    tokens = tokens{1}; % Extract matched tokens
    
    sessionNumbers(i) = str2double(tokens{1}); % Extract session number
    fileNumbers(i) = str2double(tokens{2}); % Extract file number
end

% Sort based on session number -> file number
[~, sortIdx] = sortrows([sessionNumbers', fileNumbers']);

% Reorder fileList based on sorted indices
fileList = fileList(sortIdx);

% Initialize storage for combined data using a dictionary
dataDict = containers.Map(); % Dictionary to store trials
trialOrder = {}; % To keep track of the trial order for columns
maxLength = 0; % Track the maximum number of timestamps

% Process each file
for iFile = 1:length(fileList)
    % Load the current file
    fileName = fileList(iFile).name;
    filePath = fullfile(folderPath, fileName);
    loadedData = load(filePath);

    % Extract the VideoTicks structure
    VideoTicks = loadedData.VideoTicks; % 1*n struct with 'vticks' field

    % Process each trial
    for iTrial = 1:length(VideoTicks)
        % Extract timestamps and values
        timestamps = VideoTicks(iTrial).vticks(1, :); % First row - timestamps
        values = VideoTicks(iTrial).vticks(2, :);     % Second row - values (0/non-zero)

        % Select timestamps corresponding to non-zero values
        validIndices = values ~= 0;
        filteredTimestamps = timestamps(validIndices);

        % Update maxLength
        maxLength = max(maxLength, length(filteredTimestamps));

        % Store data in dictionary: key -> 'fileName_trialIndex', value -> timestamp array
        key = sprintf('%s_Trial%d', fileName, iTrial);
        dataDict(key) = filteredTimestamps;
        trialOrder{end + 1} = key; % Store order
    end
end

% Prepare data for writing to CSV (Transpose structure)
numTrials = length(trialOrder);
outputMatrix = cell(maxLength + 2, numTrials + 1); % Allocate space for additional leftmost column

% Add leftmost column values
outputMatrix{1, 1} = 'EDFile_name';
outputMatrix{2, 1} = 'trial_index_in_file';
for i = 1:maxLength
    outputMatrix{i + 2, 1} = sprintf('timestamp%d', i);
end

% Fill the first row with file names
for iCol = 1:numTrials
    keyParts = split(trialOrder{iCol}, '_');
    outputMatrix{1, iCol + 1} = keyParts{1};  % ED file name
end

% Fill the second row with trial indices as integers
for iCol = 1:numTrials
    keyParts = split(trialOrder{iCol}, '_');
    trialIndex = erase(keyParts{2}, 'Trial');
    outputMatrix{2, iCol + 1} = int32(str2double(trialIndex));
end

% Fill the remaining rows with timestamps, padding with NaNs if needed
for iCol = 1:numTrials
    timestamps = dataDict(trialOrder{iCol});
    paddedTimestamps = nan(maxLength, 1); % Pad with NaN
    paddedTimestamps(1:length(timestamps)) = timestamps;
    outputMatrix(3:end, iCol + 1) = num2cell(paddedTimestamps);
end

% Open the file for writing
fid = fopen(outputFile, 'w');
if fid == -1
    error('Could not open file %s for writing. Check folder path or permissions.', outputFile);
end

% Write the CSV file
for iRow = 1:size(outputMatrix, 1)
    rowValues = outputMatrix(iRow, :);
    for j = 1:size(outputMatrix, 2)
        value = rowValues{j};
        
        if isnumeric(value) && isnan(value)
            fprintf(fid, 'NaN,');
        elseif isnumeric(value) && mod(value, 1) == 0  % Check if value is an integer
            fprintf(fid, '%d,', value);  % Print as integer
        elseif isnumeric(value)
            fprintf(fid, '%.5f,', value);  % Print with decimals for floats
        elseif ischar(value) || isstring(value)
            fprintf(fid, '%s,', char(value));  % Handle string values
        else
            fprintf(fid, 'NA,');  % Default case
        end
    end
    fprintf(fid, '\n');
end

% Close the file
fclose(fid);

disp(['Combined VideoTicks data exported to ', outputFile]);
