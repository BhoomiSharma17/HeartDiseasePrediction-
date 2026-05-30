#!/bin/bash

# Setup script for Heart Disease Prediction System
# This script sets up the project environment and dependencies

echo "============================================================"
echo "Heart Disease Prediction System - Setup"
echo "============================================================"

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p models
mkdir -p data
mkdir -p screenshots
mkdir -p logs

# Check if data file exists
echo ""
if [ -f "data/heart.csv" ]; then
    echo "✅ Dataset found: data/heart.csv"
else
    echo "⚠️  Dataset not found!"
    echo "Please download the dataset and place it in data/heart.csv"
    echo "Dataset: https://archive.ics.uci.edu/ml/datasets/Heart+Disease"
fi

# Train model if not exists
echo ""
if [ -f "models/best_heart_disease_model.pkl" ]; then
    echo "✅ Model found: models/best_heart_disease_model.pkl"
else
    echo "⚠️  Model not found!"
    echo "Training model... (this may take a few minutes)"
    python src/model_trainer.py
fi

echo ""
echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "To run the application:"
echo "  python run.py"
echo ""
echo "Or directly:"
echo "  streamlit run app/streamlit_app.py"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "============================================================"
