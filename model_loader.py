import torch
import torchvision.models as models
from torch.serialization import add_safe_globals
from torchvision.models.efficientnet import EfficientNet
from torch.nn.modules.container import Sequential
from torchvision.ops.misc import Conv2dNormActivation

def load_model(model_path):
    # Add all required modules to safe globals
    add_safe_globals([
        EfficientNet,
        Sequential,
        Conv2dNormActivation
    ])
    
    # Create model
    model = models.efficientnet_b0(pretrained=False)
    model.classifier = torch.nn.Linear(1280, 65)
    
    # Load weights with all necessary modules allowed
    state_dict = torch.load(
        model_path,
        map_location=torch.device('cpu'),
        weights_only=True
    )
    model.load_state_dict(state_dict)
    
    return model
