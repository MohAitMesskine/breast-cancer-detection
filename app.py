from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom model class that matches the training architecture
class BreastCancerRegNet(nn.Module):
    def __init__(self, state_dict):
        super(BreastCancerRegNet, self).__init__()
        
        # Extract the architecture from the state_dict
        # The checkpoint has: backbone.head.fc.1.weight with shape [512, 1296]
        # This means 1296 input features to the classifier
        
        # Reconstruct the original RegNet backbone using timm to match training config
        backbone = timm.create_model('regnety_064', pretrained=False)
        
        # Build the classifier to match exactly what's in the checkpoint
        # Extract dimensions from the state_dict
        fc1_in = state_dict['backbone.head.fc.1.weight'].shape[1]  # 1296
        fc1_out = state_dict['backbone.head.fc.1.weight'].shape[0]  # 512
        fc5_out = state_dict['backbone.head.fc.5.weight'].shape[0]  # 256
        fc9_out = state_dict['backbone.head.fc.9.weight'].shape[0]  # 2
        
        print(f"  Building classifier: {fc1_in} -> {fc1_out} -> {fc5_out} -> {fc9_out}")
        
        # Replace just the classifier head with correct dimensions to align with checkpoint
        backbone.head.fc = nn.Sequential(
            nn.Dropout(0.5),  # fc.0: non-param layer
            nn.Linear(fc1_in, fc1_out),  # fc.1: 1296 -> 512
            nn.ReLU(),  # fc.2
            nn.BatchNorm1d(fc1_out),  # fc.3
            nn.Dropout(0.5),  # fc.4
            nn.Linear(fc1_out, fc5_out),  # fc.5: 512 -> 256
            nn.ReLU(),  # fc.6
            nn.BatchNorm1d(fc5_out),  # fc.7
            nn.Dropout(0.5),  # fc.8
            nn.Linear(fc5_out, fc9_out)  # fc.9: 256 -> 2
        )
        
        self.backbone = backbone
    
    def forward(self, x):
        return self.backbone(x)

def load_model():
    """Load the model from checkpoint"""
    try:
        print("Loading model checkpoint...")
        checkpoint = torch.load('best_regnet_binary_model.pth', map_location=device, weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'N/A')
            val_acc = checkpoint.get('val_acc', 'N/A')
            if isinstance(val_acc, (int, float)):
                print(f"✓ Checkpoint found: Epoch {epoch}, Val Accuracy: {val_acc:.2f}%")
            else:
                print(f"✓ Checkpoint found: Epoch {epoch}")
        else:
            state_dict = checkpoint
            print("✓ State dict loaded")
        
        # Create model with custom architecture
        model = BreastCancerRegNet(state_dict)
        
        # Load weights - the state_dict has 'backbone.' prefix, so we load it directly
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"  ⚠ Missing keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"  ⚠ Unexpected keys: {len(unexpected_keys)}")
        
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("="*50)
    print("PREDICTION REQUEST RECEIVED")
    print("="*50)
    
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500
    
    print("✓ Model loaded")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    print("✓ File in request")
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    print(f"✓ Filename: {file.filename}")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file format. Please upload PNG or JPEG images'}), 400
    
    print("✓ File format valid")
    
    try:
        # Read and process the image
        print("Reading image bytes...")
        image_bytes = file.read()
        print(f"✓ Image file size: {len(image_bytes)} bytes")
        
        print(f"✓ Image file size: {len(image_bytes)} bytes")
        
        print("Opening image...")
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"✓ Image size: {image.size}, Mode: {image.mode}")
        
        # Convert image to base64 for display
        print("Converting to base64...")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        print(f"✓ Base64 string length: {len(img_str)}")
        
        # Transform image for model
        print("Transforming image...")
        img_tensor = transform(image).unsqueeze(0).to(device)
        print(f"✓ Tensor shape: {img_tensor.shape}")
        
        # Make prediction
        print("Making prediction...")
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Debug: Print output shape
            print(f"✓ Model output shape: {outputs.shape}")
            print(f"✓ Model output: {outputs}")
            
            # Check if output has the expected shape
            if outputs.shape[1] < 2:
                return jsonify({'error': f'Model output shape unexpected: {outputs.shape}. Expected 2 classes.'}), 500
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print(f"✓ Probabilities: {probabilities}")
            
            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            print(f"✓ Predicted class index: {predicted_class}")
            
            # Ensure predicted_class is within valid range
            if predicted_class >= outputs.shape[1]:
                print(f"⚠ WARNING: predicted_class {predicted_class} >= {outputs.shape[1]}, resetting to 0")
                predicted_class = 0  # Default to first class
            
            confidence = probabilities[0][predicted_class].item() * 100
            print(f"✓ Confidence: {confidence:.2f}%")
        
        # Interpret results - handle multi-class output
        # Map classes: typically 0=Benign, 1=Malignant (or first 2 classes)
        print("Interpreting results...")
        num_classes = outputs.shape[1]
        print(f"✓ Number of classes: {num_classes}")
        
        print(f"✓ Number of classes: {num_classes}")
        
        # Additional safety check
        if predicted_class < 0 or predicted_class >= num_classes:
            print(f"⚠ WARNING: Invalid predicted class {predicted_class}, resetting to 0")
            predicted_class = 0
            confidence = probabilities[0][predicted_class].item() * 100
        
        print(f"Building response for {num_classes} classes...")
        if num_classes == 2:
            class_names = ['Benign', 'Malignant']
            print(f"✓ Class names: {class_names}")
            print(f"✓ Accessing class_names[{predicted_class}]...")
            
            # Safe indexing with bounds check
            if predicted_class < len(class_names):
                prediction = class_names[predicted_class]
                print(f"✓ Prediction: {prediction}")
            else:
                print(f"⚠ WARNING: predicted_class {predicted_class} >= len(class_names) {len(class_names)}")
                prediction = class_names[0]  # Default to Benign if out of bounds
                print(f"✓ Using default: {prediction}")
            
            print("Building result dictionary...")
            result = {
                'prediction': prediction,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'Benign': round(probabilities[0][0].item() * 100, 2),
                    'Malignant': round(probabilities[0][1].item() * 100, 2)
                },
                'image': img_str
            }
            print("✓ Result built successfully")
            print(f"✓ Returning prediction: {prediction} ({confidence:.2f}%)")
            return jsonify(result)
            return jsonify(result)
        elif num_classes == 3:
            # Handle 3-class output (Benign, Malignant, Malignant_multiple)
            class_names = ['Benign', 'Malignant', 'Malignant Multiple']
            print(f"✓ 3-class mode, class_names: {class_names}")
            prediction = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
            
            # For display purposes, combine malignant classes
            benign_prob = probabilities[0][0].item() * 100
            malignant_prob = (probabilities[0][1].item() + probabilities[0][2].item()) * 100 if num_classes > 2 else probabilities[0][1].item() * 100
            
            # Simplify to binary classification for display
            if predicted_class == 0:
                simplified_prediction = 'Benign'
            else:
                simplified_prediction = 'Malignant'
            
            result = {
                'prediction': simplified_prediction,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'Benign': round(benign_prob, 2),
                    'Malignant': round(malignant_prob, 2)
                },
                'image': img_str,
                'detailed_prediction': prediction
            }
            print(f"✓ Returning 3-class prediction: {simplified_prediction}")
            return jsonify(result)
        else:
            # Handle unexpected number of classes
            print(f"⚠ ERROR: Unexpected number of classes: {num_classes}")
            return jsonify({'error': f'Unexpected number of output classes: {num_classes}'}), 500
    
    except Exception as e:
        import traceback
        import sys
        error_trace = traceback.format_exc()
        print("=" * 50)
        print("ERROR IN PREDICTION:")
        print(error_trace)
        print("=" * 50)
        # Get the specific line that failed
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb:
            print(f"Error at line {exc_tb.tb_lineno}: {exc_value}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    if model is None:
        print("⚠ Warning: Model could not be loaded. Please check the model file.")
    else:
        print("✓ Server is ready! Open http://localhost:5000 in your browser")
    app.run(debug=False, host='0.0.0.0', port=5000)
