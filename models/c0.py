"""The archived c0 stage; no alternative model implementation."""
from runtime import load_model, create_model
def build(device='cpu', pretrained=True):
    return load_model('c0', device) if pretrained else create_model('c0',7,device)
