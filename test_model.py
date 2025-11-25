import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device('cpu')

# Custom model class
class BreastCancerRegNet(nn.Module):
    def __init__(self):
        super(BreastCancerRegNet, self).__init__()
        backbone = models.regnet_y_800mf(weights=None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        self.backbone = backbone
    
    def forward(self, x):
        return self.backbone(x)

# Load checkpoint
checkpoint = torch.load('best_regnet_binary_model.pth', map_location=device, weights_only=False)
state_dict = checkpoint['model_state_dict']

print("="*50)
print("Testing CORRECT model architecture...")
print("="*50)

try:
    model = BreastCancerRegNet()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n✓ Model output shape: {output.shape}")
    print(f"  Number of classes: {output.shape[1]}")
    print(f"  Output values: {output}")
    
    # Test softmax
    probs = torch.nn.functional.softmax(output, dim=1)
    print(f"\n✓ Probabilities shape: {probs.shape}")
    print(f"  Probabilities: {probs}")
    print(f"  Sum: {probs.sum()}")
    
    predicted_class = torch.argmax(probs, dim=1).item()
    print(f"\n✓ Predicted class index: {predicted_class}")
    print(f"  Class names: {['Benign', 'Malignant']}")
    print(f"  Prediction: {['Benign', 'Malignant'][predicted_class]}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

