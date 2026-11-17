"""The archived dino_cross_stage stage; no alternative model implementation."""
from runtime import load_model, create_model
def build(device='cpu', pretrained=True):
    return load_model('dino_cross_stage', device) if pretrained else create_model('dino_cross_stage',7,device)
