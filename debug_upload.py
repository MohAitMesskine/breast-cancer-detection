import torch
import timm
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import sys

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

# Test the prediction flow
try:
    # Try to load the uploaded image from the attachment
    print("Testing prediction with ultrasound image...")
    
    # For now, use a test image
    import glob
    test_images = glob.glob('*.jpg') + glob.glob('*.png')
    if test_images:
        test_img = test_images[0]
        print(f"Using test image: {test_img}")
        
        image = Image.open(test_img).convert('RGB')
        print(f"Image loaded: {image.size}, Mode: {image.mode}")
        
        img_tensor = transform(image).unsqueeze(0)
        print(f"Tensor shape: {img_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(img_tensor)
            print(f"Model output shape: {outputs.shape}")
            print(f"Model outputs: {outputs}")
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            print(f"\nPredicted class index: {predicted_class}")
            print(f"Number of classes: {outputs.shape[1]}")
            print(f"Probabilities: {probabilities}")
            
            # Test the indexing that might be causing the error
            class_names = ['Benign', 'Malignant']
            print(f"\nClass names list: {class_names}")
            print(f"Trying to access class_names[{predicted_class}]...")
            
            if predicted_class < len(class_names):
                prediction = class_names[predicted_class]
                print(f"Success! Prediction: {prediction}")
            else:
                print(f"ERROR: predicted_class {predicted_class} >= len(class_names) {len(class_names)}")
                
    else:
        print("No test images found in directory")
        
except Exception as e:
    import traceback
    print("\n" + "="*50)
    print("ERROR:")
    print(traceback.format_exc())
    print("="*50)
