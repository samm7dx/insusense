#!/usr/bin/env bash

set -e  # stop on error

echo "🚀 Setting up InsuSense..."

# -------------------------------
# CREATE VENV (if not exists)
# -------------------------------
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# -------------------------------
# INSTALL DEPENDENCIES
# -------------------------------
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# -------------------------------
# CREATE DIRECTORIES
# -------------------------------
echo "📁 Preparing directories..."
mkdir -p model
mkdir -p data

# -------------------------------
# TRAIN MODEL (if not exists)
# -------------------------------
if [ ! -f "model/model.pkl" ]; then
    echo "🧠 Training model..."
    python train_model.py
else
    echo "✅ Model already exists. Skipping training."
fi

# -------------------------------
# DONE
# -------------------------------
echo "🎯 Setup complete!"
echo "Run the app using: streamlit run app.py"
