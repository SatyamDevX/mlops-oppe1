#!/bin/bash

# Setup
pip install -r requirements.txt
pip install scikit-learn pandas joblib

# Train model and save metrics
python train.py

# CML comment
echo "## 🎯 Model Evaluation Report" > report.md
cat metrics.txt >> report.md

# Post comment
cml comment create report.md
