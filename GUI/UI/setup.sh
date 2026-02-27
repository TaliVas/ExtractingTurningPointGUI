#!/bin/bash

# Set the relative path for the virtual environment directory
VENV_DIR="./venv"

# Check if the virtual environment already exists, if not, create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating virtual environment..."
    python3 -m venv $VENV_DIR
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating the virtual environment..."

# Check the platform (Linux/macOS or Windows)
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    # macOS/Linux
    source $VENV_DIR/bin/activate
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source $VENV_DIR/Scripts/activate
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "Virtual environment activated."

# Upgrade pip using the method suggested by the error
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo "Pip upgraded successfully."

# Install required dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "Dependencies installed successfully."

# Run the Python script
echo "Running the main script..."
python main.py
echo "Main script executed successfully."

# Deactivate the virtual environment after the script runs
echo "Deactivating the virtual environment..."
deactivate
echo "Virtual environment deactivated."
