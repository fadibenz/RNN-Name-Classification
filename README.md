# RNN Last Name Classifier

## Overview
This Jupyter notebook implements a Recurrent Neural Network (RNN) for classifying surnames by their language of origin using PyTorch. The model achieves **80.97% evaluation accuracy** after 20 epochs of training. The implementation includes data preprocessing, model architecture design, training loop, and performance visualization.

## Features
- Classifies surnames into 18 language categories
- Supports both RNN and LSTM architectures
- Implements custom data preprocessing pipeline
- Includes hyperparameter tuning experiments
- Visualizes training loss and accuracy trends
- Achieves over 80% validation accuracy

## Dataset
The dataset contains ~20k surnames from 18 languages:
- Downloaded from PyTorch tutorial data
- Includes languages like English, French, German, etc.
- Preprocessed with ASCII normalization and character encoding
- Split into 80% training (16,059 samples) and 20% test (4,015 samples)

## Model Architecture
`RecurrentClassifier` class with:
- Embedding layer
- Configurable RNN/LSTM layers
- Dropout regularization
- Linear classification head
- Hyperparameters:
  - Hidden size: 200
  - Layers: 4
  - Dropout: 0.2
  - Learning rate: 3e-4
  - Batch size: 256

## Training
- Uses AdamW optimizer
- CrossEntropyLoss for multi-class classification
- 20 epochs of training
- Batch-wise processing with progress bars
- Automatic GPU detection
- Seed configuration for reproducibility

## Results
- Final evaluation accuracy: **80.97%**
- Training loss converges smoothly
- Accuracy improves consistently across epochs
- Visualization of training dynamics included

## Hyperparameter Insights
Key findings from experimentation:
- Larger hidden layers improve performance
- Multiple RNN layers can hurt performance (vanishing gradients)
- Larger batch sizes (256) with dropout (0.2) help generalization
- Learning rate adjustments impact convergence stability

## Prerequisites
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- tqdm
- GPU recommended for training

## Usage
1. Run notebook cells sequentially
2. Data automatically downloads from PyTorch servers
3. Model trains for 20 epochs by default
4. Adjust hyperparameters in the config section
5. Visualizations generate automatically
