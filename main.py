import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
from PIL import Image
import torch
import io
from model_loader import load_model
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using EfficientNet model",
    version="1.0.0"
)

# Load your trained PyTorch model
try:
    model_path = os.environ.get('MODEL_PATH', './data_file/efficientnet_b0_plant_modelarc.pth')
    model = load_model(model_path)
    model.eval()
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Define the class labels
class_labels: List[str] = [
    'Bitter gourd - Downy Mildew', 'Bitter gourd - Healthy', 'Bitter gourd - Jassid', 'Bitter gourd - Leaf Spot',
    'Bitter gourd - Nitrogen Deficiency', 'Bitter gourd - Nitrogen and Magnesium Deficiency',
    'Bitter gourd - Nitrogen and Potassium Deficiency', 'Bitter gourd - Potassium Deficiency',
    'Bitter gourd - Potassium and Magnesium Deficiency', 'Corn_Blight', 'Corn_Common_Rust', 'Corn_Gray_Leaf_Spot',
    'Corn_Healthy', 'Cucumber_Anthracnose', 'Cucumber_Bacterial Wilt', 'Cucumber_Downy Mildew', 'Cucumber_Fresh Leaf',
    'Cucumber_Gummy Stem Blight', 'Eggplant - Epilachna Beetle', 'Eggplant - Flea Beetle', 'Eggplant - Healthy',
    'Eggplant - Jassid', 'Eggplant - Mite', 'Eggplant - Mite and Epilachna Beetle', 'Eggplant - Nitrogen Deficiency',
    'Eggplant - Nitrogen and Potassium Deficiency', 'Eggplant - Potassium Deficiency', 'Lentil_Ascochyta blight',
    'Lentil_Normal', 'Lentil_Powdery Mildew', 'Lentil_Rust', 'Paddy_bacterial_leaf_blight', 'Paddy_bacterial_leaf_streak',
    'Paddy_bacterial_panicle_blight', 'Paddy_blast', 'Paddy_brown_spot', 'Paddy_dead_heart', 'Paddy_downy_mildew',
    'Paddy_hispa', 'Paddy_normal', 'Paddy_tungro', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Sugarcane_Healthy', 'Sugarcane_Mosaic', 'Sugarcane_RedRot', 'Sugarcane_Rust', 'Sugarcane_Yellow',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
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

def validate_image(image_data: bytes) -> Image.Image:
    """Validate and open image data."""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format")

@app.get("/")
async def root() -> Dict:
    """Root endpoint returning API information."""
    return {
        "status": "active",
        "total_classes": len(class_labels),
        "classes": class_labels,
        "model_path": model_path
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Predict plant disease from uploaded image.
    
    Args:
        file: Uploaded image file
    
    Returns:
        Dictionary containing prediction, confidence score, and total classes
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and validate the image file
        image_data = await file.read()
        image = validate_image(image_data)
        
        # Preprocess the image
        image_tensor = preprocess(image).unsqueeze(0)
        
        # Make a prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get top 3 predictions
            top_3_confidences, top_3_indices = torch.topk(probabilities, 3)
            top_3_predictions = [
                {
                    "class": class_labels[idx.item()],
                    "confidence": float(conf.item())
                }
                for idx, conf in zip(top_3_indices[0], top_3_confidences[0])
            ]
        
        return {
            "primary_prediction": {
                "class": class_labels[predicted.item()],
                "confidence": float(confidence.item())
            },
            "top_3_predictions": top_3_predictions,
            "total_classes": len(class_labels)
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 2828))
    uvicorn.run(app, host="0.0.0.0", port=port)
