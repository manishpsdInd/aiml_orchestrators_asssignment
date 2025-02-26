from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
import numpy as np
from PIL import Image
import mlflow.pyfunc
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# Load model from MLflow
MODEL_URI = "runs:/4faf36d8ab8a48d1967075966a7bb631/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Initialize FastAPI
app = FastAPI()

# Serve static files and templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(await file.read())).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image_array = np.array(image).astype(np.float32) / 255.0  # Normalize
        image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model
        
        # Perform inference
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        
        return {"digit": int(predicted_digit)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

