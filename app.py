from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
from pathlib import Path
from io import BytesIO
import cv2

app = FastAPI()

# Load the model and class names when the application starts
@app.on_event("startup")
async def load_model():
    global model
    global categories
    global __model
    global __classes

    with open('categories.json') as json_file:
        categories = json.load(json_file)

    f = Path("model_structure.json")
    model_structure = f.read_text()
    model = model_from_json(model_structure)
    model.load_weights("model_weights.h5")

    __model = model
    __classes = categories

def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    img_array = np.array(img)
    images = np.expand_dims(img_array, axis=0)
    images = preprocess_input(images)
    return images

def predict_image_class(images):
    results = __model.predict(images)
    class_index = np.argmax(results)
    max_probability = np.round(np.max(results) * 100, 2)
    predicted_class = __classes[class_index]
    
    if (predicted_class == "guppy fish" and max_probability < 80.0) or (predicted_class == "arowana" and max_probability < 70.0) or (predicted_class == "gold fish" and max_probability < 60.0):
        return "unknown", max_probability
    elif max_probability < 50.0:
        return "unknown", max_probability
    else:
        return predicted_class, max_probability

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = image.load_img(BytesIO(contents), target_size=(256, 256))
    img_array = image.img_to_array(img)
    images = np.expand_dims(img_array, axis=0)
    images = preprocess_input(images)

    predicted_class, max_probability = predict_image_class(images)

    return JSONResponse(content={"predicted_class": predicted_class, "confidence_score": max_probability})
