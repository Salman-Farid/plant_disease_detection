import torch
import os
import requests

def load_model(model_path):
    if os.path.exists(model_path):
        return torch.load(model_path, map_location=torch.device('cpu'))
    else:
        # If the model doesn't exist locally, download it
        url = os.environ.get('MODEL_URL')
        if url:
            response = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            return torch.load(model_path, map_location=torch.device('cpu'))
        else:
            raise FileNotFoundError(f"Model not found at {model_path} and MODEL_URL not set")

