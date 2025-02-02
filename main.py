from typing import Dict, List, Any
import os

from fastapi import FastAPI, File, UploadFile
from sympy import Dict
from torchvision import transforms
from PIL import Image
import torch
import io

app = FastAPI()

# Load your trained PyTorch model
model_path = os.environ.get('MODEL_PATH', './data_file/efficientnet_b0_plant_modelarc.pth')
model = torch.load(model_path, map_location=torch.device('cpu'))  # Use 'cuda' if you have a GPU
model.eval()

# Define the class labels
class_labels = [
    'Bitter gourd - Downy Mildew', 'Bitter gourd - Healthy', 'Bitter gourd - Jassid', 'Bitter gourd - Leaf Spot',
    'Bitter gourd - Nitrogen Deficiency', 'Bitter gourd - Nitrogen and Magnesium Deficiency',
    'Bitter gourd - Nitrogen and Potassium Deficiency', 'Bitter gourd - Potassium Deficiency',
    'Bitter gourd - Potassium and Magnesium Deficiency', 'Corn_Blight', 'Corn_Common_Rust', 'Corn_Gray_Leaf_Spot',
    'Corn_Healthy', 'Cucumber_Anthracnose', 'Cucumber_Bacterial Wilt', 'Cucumber_Downy Mildew', 'Cucumber_Fresh Leaf',
    'Cucumber_Gummy Stem Blight', 'Eggplant - Epilachna Beetle', 'Eggplant - Flea Beetle', 'Eggplant - Healthy',
    'Eggplant - Jassid', 'Eggplant - Mite', 'Eggplant - Mite and Epilachna Beetle', 'Eggplant - Nitrogen Deficiency',
    'Eggplant - Nitrogen and Potassium Deficiency', 'Eggplant - Potassium Deficiency', 'Lentil_Ascochyta blight',
    'Lentil_Normal', 'Lentil_Powdery Mildew', 'Lentil_Rust', 'Paddy_bacterial_leaf_blight',
    'Paddy_bacterial_leaf_streak',
    'Paddy_bacterial_panicle_blight', 'Paddy_blast', 'Paddy_brown_spot', 'Paddy_dead_heart', 'Paddy_downy_mildew',
    'Paddy_hispa', 'Paddy_normal', 'Paddy_tungro', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Sugarcane_Healthy', 'Sugarcane_Mosaic', 'Sugarcane_RedRot', 'Sugarcane_Rust', 'Sugarcane_Yellow',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy', 'Wheat_Brown rust', 'Wheat_Healthy', 'Wheat_Loose Smut',
    'Wheat_Mildew', 'Wheat_Septoria', 'Wheat_Stem Rust', 'Wheat_Yellow rust'
]

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/")
async def root() -> dict[str, int | str | list[str | Any]]:
    """Root endpoint returning API information."""
    return {
        "total_classes": len(class_labels),
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Preprocess the image
    image = preprocess(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = class_labels[predicted.item()]

    return {"prediction": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
