import torch
import torchvision.models as models
from torch.serialization import add_safe_globals
from torchvision.models.efficientnet import EfficientNet

def load_model(model_path):
    # Allow EfficientNet for safe deserialization
    add_safe_globals([EfficientNet])


    # Create model
    model = models.efficientnet_b0(pretrained=False)
    model.classifier = torch.nn.Linear(1280, 65)

    # Load weights with weights_only=True
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)

    return model

def verify_model_loading(model_path):
    """
    Utility function to verify model loading
    """
    try:
        return load_model(model_path)
    except Exception:
        return None
