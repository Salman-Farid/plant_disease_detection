import torch
import torchvision.models as models
from torch.serialization import add_safe_globals
from torchvision.models.efficientnet import EfficientNet
from torch.nn.modules.container import Sequential

def load_model(model_path):
    # Add both EfficientNet and Sequential to safe globals
    add_safe_globals([EfficientNet, Sequential])
    
    # Create a new EfficientNet model
    model = models.efficientnet_b0(pretrained=False)
    
    # Modify the classifier for your number of classes (65 classes)
    model.classifier = torch.nn.Linear(1280, 65)
    
    # Load the state dict with weights_only=True
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    
    return model
