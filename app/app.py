import streamlit as st
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import os
from PIL import Image

# Path to stored models
MODEL_PATH = os.path.abspath(os.path.join(os.getcwd(), "../models"))

# Load available models
def get_model_list():
    return [f for f in os.listdir(MODEL_PATH) if f.endswith(".h5") or f.endswith(".pth")]

# Load and preprocess image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST dimensions
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input (TensorFlow)
    return image

# Define PyTorch CNN Model
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

# Predict function for TensorFlow
def predict_image_tf(model, image):
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    return predicted_label

# Predict function for PyTorch
def predict_image_torch(model, image):
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)  # Reshape for PyTorch
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit and select a model to classify it.")

# Model selection
dropdown_models = get_model_list()
selected_model = st.selectbox("Select a trained model", dropdown_models)

# Load selected model
if selected_model:
    if selected_model.endswith(".h5"):
        model = tf.keras.models.load_model(os.path.join(MODEL_PATH, selected_model))
        model_type = "tensorflow"
        st.success(f"Loaded TensorFlow model: {selected_model}")
    elif selected_model.endswith(".pth"):
        model = CNN()
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, selected_model)))
        model.eval()
        model_type = "pytorch"
        st.success(f"Loaded PyTorch model: {selected_model}")

# Image upload
uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(image)
    
    # Predict and display result
    if st.button("Classify Image"):
        if model_type == "tensorflow":
            prediction = predict_image_tf(model, processed_image)
        else:
            prediction = predict_image_torch(model, processed_image)
        st.success(f"Predicted Digit: {prediction}")
