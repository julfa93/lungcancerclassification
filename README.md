# Lung Cancer Classification using Deep Learning Models

This repository contains a deep learning project aimed at classifying lung cancer images into three categories: **benign**, **malignant**, and **normal**. The classification is performed using an ensemble of pre-trained models: **EfficientNetV2B0**, **ResNet50V2**, and **DenseNet121**, combined into a single ensemble model to enhance performance.

## Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early and accurate detection is crucial for improving survival rates. In this project, we leverage state-of-the-art deep learning models for image classification to distinguish between different types of lung cancer and normal lung images.

The models were trained on [The IQ-OTHNCCD lung cancer dataset](doi: 10.17632/bhmdr45bh2.4), and the best-performing individual models were combined into an ensemble model for improved accuracy.

## Dataset

The dataset used in this project is publicly available on Kaggle:

- **Dataset Name**: The IQ-OTHNCCD lung cancer dataset
- **Categories**: Benign, Malignant, Normal
- **Input Image Size**: 150x150 pixels (resized for training)
  
The dataset includes labeled chest X-ray images categorized into three classes:
- **Benign cases**: Images indicating benign tumors
- **Malignant cases**: Images indicating malignant tumors
- **Normal cases**: Images of healthy lungs

## Models Used

Three pre-trained models from the **TensorFlow/Keras** library were fine-tuned on this dataset:
1. **EfficientNetV2B0**
2. **ResNet50V2**
3. **DenseNet121**

After training, these models were combined into an ensemble using average pooling of their predictions to improve classification accuracy. The ensemble model leverages the strengths of each individual model.

## Project Structure

The repository is organized as follows:

```
├── dataset/                   # Folder containing dataset (if needed locally)
├── models/                    # Folder containing saved models
├── notebooks/                 # Jupyter notebooks for training, evaluation, and visualization
├── src/                       # Python scripts for data preprocessing, model training, and evaluation
│   ├── train.py               # Main script for training models
│   ├── evaluate.py            # Script to evaluate and plot confusion matrices
│   ├── ensemble.py            # Code for building and training the ensemble model
│   └── utils.py               # Utility functions
├── logs/                      # Folder containing TensorBoard logs
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files to be ignored by Git
```

## How to Run the Project

### Prerequisites

1. Install Python 3.x (if not already installed).
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/lung-cancer-classification.git
   cd lung-cancer-classification
   ```

3. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Models

1. To train individual models (EfficientNetV2B0, ResNet50V2, DenseNet121) and save the models, run:
   ```bash
   python src/train.py
   ```

2. To build and train the ensemble model:
   ```bash
   python src/ensemble.py
   ```

### Evaluating the Models

After training, evaluate the models using the test set and plot confusion matrices:
```bash
python src/evaluate.py
```

### Predicting on New Images

To predict the class of a new lung X-ray image:
1. Run the Jupyter notebook for predictions or use the command-line script to upload and classify images.
2. Alternatively, use `src/utils.py` for a custom script that predicts an image.


## Future Improvements

1. Implementing techniques like Grad-CAM for better understanding of model predictions.
2. Use more dataset 

