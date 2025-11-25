import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cpu')

# Load checkpoint and inspect structure
checkpoint = torch.load('best_regnet_binary_model.pth', map_location=device, weights_only=False)
state_dict = checkpoint['model_state_dict']

print("Analyzing model structure from checkpoint...")
print("="*60)

# Check backbone keys
backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]
print(f"\nBackbone keys: {len(backbone_keys)}")
print("Sample backbone keys:")
for key in list(backbone_keys)[:5]:
    print(f"  {key}: {state_dict[key].shape}")

# Check head/fc keys
fc_keys = [k for k in state_dict.keys() if 'head.fc' in k]
print(f"\nHead FC keys: {len(fc_keys)}")
print("All head.fc keys:")
for key in fc_keys:
    print(f"  {key}: {state_dict[key].shape}")

# Determine the exact architecture
print("\n" + "="*60)
print("Reconstructing architecture...")
print("="*60)

# Create base model to compare
base_model = models.regnet_y_800mf(weights=None)
print(f"\nBase RegNet fc input features: {base_model.fc.in_features}")
print(f"Base RegNet fc output features: {base_model.fc.out_features}")

# Check the structure of saved fc layers
print("\nSaved model FC structure:")
fc_layers = {}
for k in sorted(fc_keys):
    layer_num = k.split('.')[-2]  # Get the layer number
    param_type = k.split('.')[-1]  # weight, bias, etc.
    if layer_num not in fc_layers:
        fc_layers[layer_num] = {}
    fc_layers[layer_num][param_type] = state_dict[k].shape

for layer, params in sorted(fc_layers.items()):
    print(f"  Layer {layer}: {params}")
