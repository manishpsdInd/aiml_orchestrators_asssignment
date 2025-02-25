import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import streamlit as st
import matplotlib.pyplot as plt

# MLflow setup
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Change if using remote MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load TensorFlow Model
tensorflow_model = mlflow.tensorflow.load_model("runs:/f543ff7b860649c0a78b61b5d05df2d2/model")

# Load PyTorch Model
pytorch_model_path = mlflow.pytorch.load_model("runs:/4faf36d8ab8a48d1967075966a7bb631/model")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load PyTorch Model
pytorch_model = CNN()
pytorch_model.load_state_dict(pytorch_model_path.state_dict())
pytorch_model.eval()

# Function to preprocess input image
def preprocess_image(image):
    image = image.resize((28, 28))
    image = ImageOps.grayscale(image)
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Function to predict using TensorFlow Model
def predict_with_tensorflow(image):
    image = preprocess_image(image)
    pred = tensorflow_model.predict(image)
    return np.array(pred[0])

# Function to predict using PyTorch Model
def predict_with_pytorch(image):
    image = preprocess_image(image)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        pred = torch.nn.functional.softmax(pytorch_model(image), dim=1)
    return pred.numpy()[0]

# Function to ensemble predictions
def ensemble_prediction(image):
    tf_pred = predict_with_tensorflow(image)
    pt_pred = predict_with_pytorch(image)
    final_pred = (tf_pred + pt_pred) / 2
    return np.argmax(final_pred), final_pred

# Streamlit App UI
st.title("📝 Handwritten Digit Recognition - Ensemble Model")
st.write("Upload an image of a handwritten digit (0-9) to recognize.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    digit, probabilities = ensemble_prediction(image)
    
    # Show Results
    st.write(f"### 🏆 Predicted Digit: {digit}")
    fig, ax = plt.subplots()
    ax.bar(range(10), probabilities)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digits")
    ax.set_ylabel("Confidence")
    st.pyplot(fig)

