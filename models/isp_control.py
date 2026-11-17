"""The archived isp_control stage; no alternative model implementation."""
from runtime import load_model, create_model
def build(device='cpu', pretrained=True):
    return load_model('isp_control', device) if pretrained else create_model('isp_control',7,device)
