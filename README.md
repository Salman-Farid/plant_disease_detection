# ![FastAPI Logo](https://github.com/Salman-Farid/plant_disease_detection/blob/main/fast_api.png)
# ![Vercel Logo](https://github.com/Salman-Farid/plant_disease_detection/blob/main/vercel.png)

```markdown
# Plant Disease Detection API

This FastAPI-based API serves a quantized PyTorch model (EfficientNet-B0) for plant disease detection. The model can classify images of plants into 66 different classes, helping users identify diseases and conditions in their plants.

## 🚀 Getting Started

To get started with the Plant Disease Detection API, follow the steps below to set up and run the server.

### 1. Clone the Repository

```bash
git clone https://github.com/Salman-Farid/plant_disease_detection.git
cd plant_disease_detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Before running the server, ensure you have the `MODEL_PATH` environment variable set, which points to your trained model.

```bash
export MODEL_PATH="./data_file/efficientnet_b0_plant_modelarc.pth"
```

### 4. Run the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 2828
```

The API server will now be running at [http://localhost:2828](http://localhost:2828).

---

## 🧑‍🔬 API Endpoints

### 1. **POST** `/predict/`

This endpoint accepts an image file and returns a prediction for the plant disease.

#### Request

- **Method**: POST
- **Endpoint**: `/predict/`
- **Parameters**: 
    - **file**: A plant image (JPEG/PNG format) for diagnosis.

#### Example Request

```bash
curl -X 'POST' \
  'http://localhost:2828/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_image_path.jpg'
```

#### Response

- **Prediction**: Returns a string representing the plant disease or condition.

```json
{
  "prediction": "Bitter gourd - Downy Mildew"
}
```

---

## 📄 Model Details

- **Model Type**: EfficientNet-B0 (Quantized)
- **Class Labels**: The model classifies images into the following plant diseases and conditions:
  
  ```
  Bitter gourd - Downy Mildew, Bitter gourd - Healthy, Bitter gourd - Jassid, Bitter gourd - Leaf Spot,
  Bitter gourd - Nitrogen Deficiency, Bitter gourd - Nitrogen and Magnesium Deficiency,
  Bitter gourd - Nitrogen and Potassium Deficiency, Bitter gourd - Potassium Deficiency,
  Bitter gourd - Potassium and Magnesium Deficiency, Corn Blight, Corn Common Rust, Corn Gray Leaf Spot,
  Corn Healthy, Cucumber Anthracnose, Cucumber Bacterial Wilt, Cucumber Downy Mildew, Cucumber Fresh Leaf,
  Cucumber Gummy Stem Blight, Eggplant - Epilachna Beetle, Eggplant - Flea Beetle, Eggplant - Healthy,
  Eggplant - Jassid, Eggplant - Mite, Eggplant - Mite and Epilachna Beetle, Eggplant - Nitrogen Deficiency,
  Eggplant - Nitrogen and Potassium Deficiency, Eggplant - Potassium Deficiency, Lentil Ascochyta Blight,
  Lentil Normal, Lentil Powdery Mildew, Lentil Rust, Paddy Bacterial Leaf Blight, Paddy Bacterial Leaf Streak,
  Paddy Bacterial Panicle Blight, Paddy Blast, Paddy Brown Spot, Paddy Dead Heart, Paddy Downy Mildew,
  Paddy Hispa, Paddy Normal, Paddy Tungro, Potato Early Blight, Potato Late Blight, Potato Healthy,
  Sugarcane Healthy, Sugarcane Mosaic, Sugarcane RedRot, Sugarcane Rust, Sugarcane Yellow,
  Tomato Bacterial Spot, Tomato Early Blight, Tomato Late Blight, Tomato Leaf Mold, Tomato Septoria Leaf Spot,
  Tomato Spider Mites, Tomato Target Spot, Tomato Tomato YellowLeaf Curl Virus,
  Tomato Tomato Mosaic Virus, Tomato Healthy, Wheat Brown Rust, Wheat Healthy, Wheat Loose Smut,
  Wheat Mildew, Wheat Septoria, Wheat Stem Rust, Wheat Yellow Rust
  ```

---

## 🛠️ Technology Used

- 🚀 **[PyTorch](https://pytorch.org/)** – For providing efficient tools to deploy ML models.  
- 📖 **[Flutter Documentation](https://flutter.dev/docs)** – For comprehensive development resources.  
- 🔐 **[Firebase](https://firebase.google.com/)** – For secure authentication and backend support.  
- ⚡ **[FastAPI](https://fastapi.tiangolo.com/)** – For serving the ML model efficiently.  
- 🌐 **[Vercel](https://vercel.com/)** – For deploying the model API server and making it globally accessible.  

---

## 📱 PlantTreatmonty App

The PlantTreatmonty mobile application utilizes this API to provide plant disease detection features directly to your device. You can use it to easily diagnose plant diseases with just an image.

**GitHub Repository**: [PlantTreatmonty Flutter App](https://github.com/Salman-Farid/planty)

### Features of the App:
- **Plant Disease Detection**: Using the model served by this API, users can upload images to get fast and accurate predictions.
- **Firebase Authentication**: Secure login and user management.
- **Plant Care Tips**: Discover helpful tips for plant care.
- **Responsive Design**: Optimized for various screen sizes for an improved user experience.
- **Community Feedback**: Users can share their experiences and provide feedback to improve the plant disease detection system and overall app functionality.

---

## 💻 Source Code

- **API Server**: [GitHub - Plant Disease Detection API](https://github.com/Salman-Farid/plant_disease_detection)
- **PlantTreatmonty App**: [GitHub - PlantTreatmonty Flutter App](https://github.com/Salman-Farid/planty)

---

##  Acknowledgments

- **PyTorch** for providing tools to deploy efficient machine learning models.
- **FastAPI** for building the API to serve the model.
- **Flutter Documentation** for helping create the mobile app.
- **Firebase** for the user authentication and backend services.
- **🌐 Vercel** – For deploying the model API server.

---

For more detailed instructions, check the official [PlantTreatmonty GitHub Repository](https://github.com/Salman-Farid/planty) or visit the [API Documentation](https://plant-disease-detection-2-aa5x.onrender.com/docs#/default/predict_predict__post).
``` 
