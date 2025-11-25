import torch
import timm
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os

# Load checkpoint
ckpt = torch.load('best_regnet_binary_model.pth', map_location='cpu', weights_only=False)
state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

# Build model
class BreastCancerRegNet(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        backbone = timm.create_model('regnety_064', pretrained=False)
        fc1_in = state_dict['backbone.head.fc.1.weight'].shape[1]
        fc1_out = state_dict['backbone.head.fc.1.weight'].shape[0]
        fc5_out = state_dict['backbone.head.fc.5.weight'].shape[0]
        fc9_out = state_dict['backbone.head.fc.9.weight'].shape[0]
        backbone.head.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc1_in, fc1_out),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_out),
            nn.Dropout(0.5),
            nn.Linear(fc1_out, fc5_out),
            nn.ReLU(),
            nn.BatchNorm1d(fc5_out),
            nn.Dropout(0.5),
            nn.Linear(fc5_out, fc9_out)
        )
        self.backbone = backbone
    def forward(self, x):
        return self.backbone(x)

model = BreastCancerRegNet(state_dict)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test with actual image
test_image = 'test_malignant.jpg'
if os.path.exists(test_image):
    image = Image.open(test_image).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        print(f'Output shape: {output.shape}')
        print(f'Output: {output}')
        
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        
        class_names = ['Benign', 'Malignant']
        print(f'Predicted: {class_names[predicted]}')
        print(f'Confidence: {probs[0][predicted].item() * 100:.2f}%')
        print(f'Benign: {probs[0][0].item() * 100:.2f}%')
        print(f'Malignant: {probs[0][1].item() * 100:.2f}%')
else:
    print(f'{test_image} not found')
