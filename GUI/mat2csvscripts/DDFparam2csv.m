% Get a list of all .mat files in the 'Info' folder
matFiles = dir('Info/*.mat');

% Check if any .mat files exist
if isempty(matFiles)
    error('No .mat files found in the Info folder.');
else
    % Load the first .mat file found in the 'Info' folder
    matFilePath = fullfile('Info', matFiles(1).name);  % Full path to the file
    load(matFilePath);  % Load the file

    % Select the DDFparam structure (assuming it exists in the file)
    if exist('DDFparam', 'var')
        data = DDFparam;  
    else
        error('DDFparam structure not found in the loaded .mat file.');
    end
end

% Set the output file path to the Info folder
outputFolder = 'Info';
csvFileName = fullfile(outputFolder, 'DDFparam.csv'); % Combine folder and file name

% Open a file for writing
fid = fopen(csvFileName, 'w');

% Collect all top-level headers
mainHeaders = fieldnames(data);
nestedHeaders = fieldnames(data(1).Electrode); % Assuming all Electrode structs have the same fields

% Exclude specific columns
columnsToExclude = {'MATDir', 'MAPDir', 'Version', 'VersionCmnt'};
mainHeaders = setdiff(mainHeaders, columnsToExclude, 'stable'); % Remove excluded columns

% Combine headers
headers = [mainHeaders; nestedHeaders];
fprintf(fid, '%s,', headers{1:end-1}); % Write all headers except the last one with a comma
fprintf(fid, '%s\n', headers{end});    % Write the last header without a trailing comma

% Write data
for i = 1:numel(data)
    % Prepare the mainRow for top-level fields
    mainRow = cell(1, numel(mainHeaders));
    for j = 1:numel(mainHeaders)
        value = data(i).(mainHeaders{j});
        if isnumeric(value) || islogical(value)
            mainRow{j} = num2str(value);
        elseif ischar(value)
            mainRow{j} = value;
        else
            mainRow{j} = 'NA'; % Placeholder for unsupported data types
        end
    end
    
    % Write rows from Electrode field
    for k = 1:numel(data(i).Electrode)
        nestedRow = cell(1, numel(nestedHeaders));
        for m = 1:numel(nestedHeaders)
            nestedValue = data(i).Electrode(k).(nestedHeaders{m});
            if isnumeric(nestedValue) || islogical(nestedValue)
                nestedRow{m} = num2str(nestedValue);
            elseif ischar(nestedValue)
                nestedRow{m} = nestedValue;
            else
                nestedRow{m} = 'NA'; % Placeholder for unsupported data types
            end
        end
        % Combine the mainRow and nestedRow into a full row
        fullRow = [mainRow, nestedRow];
        fprintf(fid, '%s,', fullRow{1:end-1}); % Write all values except the last one with a comma
        fprintf(fid, '%s\n', fullRow{end});    % Write the last value without a trailing comma
    end
end

% Close the file
fclose(fid);

disp(['DDFparam expanded with Electrode data exported to ' csvFileName]);
