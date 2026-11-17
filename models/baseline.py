"""The archived baseline stage; no alternative model implementation."""
from runtime import load_model, create_model
def build(device='cpu', pretrained=True):
    return load_model('baseline', device) if pretrained else create_model('baseline',7,device)
