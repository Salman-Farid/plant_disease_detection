import torch
import torchvision.models as models
from torch.nn import AdaptiveAvgPool2d
from torch.serialization import add_safe_globals, safe_globals
from torchvision.models.efficientnet import EfficientNet
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU, ReLU
from torchvision.ops.misc import Conv2dNormActivation
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
import torch.nn.modules.adaptive

def load_model(model_path):
    # Add all required modules to safe globals for EfficientNet
    required_modules = [
        EfficientNet,
        Sequential,
        Conv2dNormActivation,
        Conv2d,
        BatchNorm2d,
        SiLU,
        ReLU,
        Dropout,
        Linear,
        AdaptiveAvgPool2d
    ]

    # Use context manager to ensure safe loading
    with safe_globals(required_modules):
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

def verify_model_loading(model_path):
    """
    Utility function to verify model loading and print model structure
    """
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        print("\nModel structure:")
        print(model)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
