import os
import sys
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# MLflow setup
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_URI = "runs:/4faf36d8ab8a48d1967075966a7bb631/model"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model from MLflow
loaded_model = mlflow.pyfunc.load_model(MODEL_URI)

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Convert to (1, 1, 28, 28)
    return image

# Streamlit UI
st.title("📝 Handwritten Digit Recognition - MLflow Model")
st.write("Upload an image of a handwritten digit (0-9) to recognize.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = loaded_model.predict(processed_image.numpy())
    
    # Display result
    st.write(f"### 🏆 Predicted Digit: {np.argmax(prediction)}")
    
    # Show probabilities
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_xticks(range(10))
    ax.set_xlabel("Digits")
    ax.set_ylabel("Confidence")
    st.pyplot(fig)
