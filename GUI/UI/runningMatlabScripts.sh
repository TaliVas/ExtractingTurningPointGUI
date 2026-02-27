#!/bin/bash

# Prompt user for MATLAB scripts folder
read -rp "Enter the folder path containing MATLAB scripts: " SCRIPT_FOLDER

# Check if the folder exists
if [ ! -d "$SCRIPT_FOLDER" ]; then
    echo "Error: Folder '$SCRIPT_FOLDER' does not exist."
    exit 1
fi

# Prompt user for the data folder (same for all scripts)
read -rp "Enter the data folder path (used for all scripts): " DATA_FOLDER

# Check if the data folder exists
if [ ! -d "$DATA_FOLDER" ]; then
    echo "Error: Data folder '$DATA_FOLDER' does not exist."
    exit 1
fi

# Find all MATLAB scripts in the given folder
MATLAB_SCRIPTS=($(find "$SCRIPT_FOLDER" -maxdepth 1 -type f -name "*.m"))

# Check if any MATLAB scripts were found
if [ ${#MATLAB_SCRIPTS[@]} -eq 0 ]; then
    echo "No MATLAB scripts found in '$SCRIPT_FOLDER'."
    exit 1
fi

# Execute each MATLAB script with the same data folder
for script in "${MATLAB_SCRIPTS[@]}"; do
    script_name=$(basename "$script" .m)
   
    echo "Running $script_name.m with data folder: $DATA_FOLDER ..."
   
    # Run the MATLAB script, passing the data folder as an argument
    matlab -nodisplay -nosplash -r "$script_name('$DATA_FOLDER'); exit"
   
    # Check if MATLAB execution was successful
    if [ $? -ne 0 ]; then
        echo "Error executing $script_name.m"
        exit 1
    fi
done

echo "All scripts executed successfully."