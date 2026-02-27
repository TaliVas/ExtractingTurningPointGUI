% Get a list of all .mat files in the 'Info' folder
matFiles = dir('Info/*.mat');

% Check if any .mat files exist
if isempty(matFiles)
    error('No .mat files found in the Info folder.');
else
    % Load the first .mat file found in the 'Info' folder
    matFilePath = fullfile('Info', matFiles(1).name);  % Full path to the file
    load(matFilePath);  % Load the file

    if exist('SESSparam', 'var')
        data = SESSparam;  
    else
        error('SESSparam structure not found in the loaded .mat file.');
    end
end

% Select the DDFparam structure
data = DDFparam;

% Set the output file path to the Info folder
outputFolder = 'Info';
csvFileName = fullfile(outputFolder, 'SESSparam.csv'); % Combine folder and file name

% Open a file for writing
fid = fopen(csvFileName, 'w');

% Load headers for both files
fileConfigHeaders = fieldnames(SESSparam.fileConfig);
subSessHeaders = {'Files', 'TaskMode', 'Electrode1Depth', 'Electrode2Depth', 'Electrode3Depth', 'Electrode4Depth'};
combinedHeaders = [{'Subsession'}, fileConfigHeaders', subSessHeaders];

% Write headers
fprintf(fid, '%s,', combinedHeaders{1:end-1}); % Write all headers except the last one with a comma
fprintf(fid, '%s\n', combinedHeaders{end});    % Write the last header without a trailing comma

% Iterate through SubSess data
fileIndex = 1; % Initialize file index
for iSubSess = 1:numel(SESSparam.SubSess)
    subSessRow = SESSparam.SubSess(iSubSess);

    % Extract the TaskMode, Files, and Electrode Depths
    filesRange = sprintf('%d-%d', subSessRow.Files(1), subSessRow.Files(2));
    taskMode = subSessRow.TaskMode;
    electrodeDepths = cell(1, 4);
    for eIdx = 1:4
        if isfield(subSessRow.Electrode(eIdx), 'Depth')
            electrodeDepths{eIdx} = num2str(subSessRow.Electrode(eIdx).Depth);
        else
            electrodeDepths{eIdx} = 'NA'; % If Depth is missing
        end
    end

    % Get the file range for the current SubSess
    startFile = subSessRow.Files(1);
    endFile = subSessRow.Files(2);

    % Repeat SubSess data for each file in the current range
    for iFile = startFile:endFile
        % Extract the corresponding fileConfig row
        fileConfigRow = SESSparam.fileConfig(fileIndex);
        fileConfigData = cell(1, numel(fileConfigHeaders));
        for iHeader = 1:numel(fileConfigHeaders)
            value = fileConfigRow.(fileConfigHeaders{iHeader});
            if isnumeric(value) || islogical(value)
                fileConfigData{iHeader} = num2str(value);
            elseif ischar(value)
                fileConfigData{iHeader} = value;
            else
                fileConfigData{iHeader} = 'NA'; % Placeholder for unsupported types
            end
        end

        % Combine SubSess and fileConfig data
        rowData = [{sprintf('%d', iSubSess)}, fileConfigData, {filesRange, taskMode}, electrodeDepths];

        % Write the row to the file
        fprintf(fid, '%s,', rowData{1:end-1}); % Write all values except the last one with a comma
        fprintf(fid, '%s\n', rowData{end});    % Write the last value without a trailing comma

        % Increment the file index
        fileIndex = fileIndex + 1;
    end
end

% Close the file
fclose(fid);

disp(['Combined file exported to ' csvFileName]);
