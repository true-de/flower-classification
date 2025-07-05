# 🌼 Blossom AI: Flower Classification Project

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

## 📋 Overview

Blossom AI is a deep learning project that classifies five types of flowers (daisy, dandelion, rose, sunflower, and tulip) using a convolutional neural network. The project includes both a model training script and an interactive web application for real-time flower classification.

![Blossom AI App](https://raw.githubusercontent.com/username/flower-classification/main/app_screenshot.png)

## ✨ Features

- **Advanced CNN Architecture**: Utilizes a deep CNN with residual connections for improved accuracy
- **Data Augmentation**: Implements comprehensive image augmentation techniques to enhance model robustness
- **Interactive Web UI**: Beautiful Streamlit interface with modern design elements
- **Detailed Analysis**: Provides confidence scores, color analysis, and image properties for each prediction
- **Model Performance Metrics**: Visualizes training history and confusion matrix
- **Prediction History**: Keeps track of previous predictions during the session

## 🌸 Flower Classes

- 🌼 **Daisy**: Simple white petals with yellow center
- 🌻 **Dandelion**: Bright yellow composite flower
- 🌹 **Rose**: Classic layered petals, often red or pink
- 🌻 **Sunflower**: Large yellow petals with dark center
- 🌷 **Tulip**: Cup-shaped flower with smooth petals

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/flower-classification.git
   cd flower-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Training the Model

To train the flower classification model:

```bash
python train_flower_classifier.py
```

This will:
- Load and preprocess the flower dataset
- Apply data augmentation
- Train the CNN model
- Save the trained model as `flower_classifier.h5`
- Generate performance visualizations

### Running the Web Application

To launch the interactive web application:

```bash
streamlit run app.py
```

This will start a local web server, and you can access the application in your browser.

## 📊 Model Architecture

The model uses a CNN architecture with:
- Multiple convolutional blocks with batch normalization
- Residual connections for better gradient flow
- Dropout layers to prevent overfitting
- Global average pooling
- Dense layers with ReLU activation

## 📁 Project Structure

```
├── app.py                   # Streamlit web application
├── train_flower_classifier.py  # Model training script
├── flower_classifier.h5     # Trained model file
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── confusion_matrix.png     # Confusion matrix visualization
├── class_confidence.png     # Class confidence visualization
├── training_history.png     # Training history plot
├── model_metrics.json       # Model performance metrics
├── training_history.json    # Training history data
└── flowers/                 # Dataset directory
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/
```

## 📈 Performance

The model achieves approximately 85% accuracy on the validation set. Detailed metrics including precision, recall, and F1-score for each class are saved in the `model_metrics.json` file.

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.0+
- Streamlit
- NumPy
- Pillow
- scikit-learn
- Matplotlib
- Seaborn

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The dataset used is based on the [Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition) dataset from Kaggle
- Thanks to the TensorFlow and Streamlit teams for their amazing frameworks