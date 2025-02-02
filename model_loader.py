import torch
import torchvision.models as models
from torch.serialization import add_safe_globals
from torchvision.models.efficientnet import EfficientNet

def load_model(model_path):
    # Add EfficientNet to safe globals
    add_safe_globals([EfficientNet])
    
    # Create a new EfficientNet model
    model = models.efficientnet_b0(pretrained=False)
    
    # Modify the classifier for your number of classes (65 classes)
    model.classifier = torch.nn.Linear(1280, 65)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model
