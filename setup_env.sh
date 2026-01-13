#!/bin/bash

# chmod +x setup_env.sh
# ./setup_env.sh

# Set up error handling
set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up Python virtual environment..."

# Create virtual environment
python3 -m venv ~/env

# Activate virtual environment
source ~/env/bin/activate

# Update pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools

# Install requirements
echo "Installing requirements"
pip install -r requirements.txt -U

# Install torch and torchvision
echo "Installing torch and torchvision..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -U
pip install numpy==1.26.4

# List installed packages for verification
echo -e "\nInstalled packages:"
pip list

# Instructions for future use
echo -e "\n======================="
echo "Setup complete!"
echo "======================="
echo "To activate this environment in the future, run:"
echo "    source ~/env/bin/activate"
echo ""
echo "To deactivate the environment, simply run:"
echo "    deactivate"
echo ""
echo "Your virtual environment is located at: ~/env"
echo "======================="

