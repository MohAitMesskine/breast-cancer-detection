# Breast Cancer Detection AI

A Flask web application that uses deep learning (RegNet model) to detect breast cancer from medical images. The application provides a modern, user-friendly interface for uploading images and receiving instant AI-powered predictions.

## Features

- 🧠 **Advanced AI Model**: Uses RegNet-Y 400MF architecture for accurate predictions
- 🎨 **Modern UI**: Beautiful, responsive design with gradient backgrounds
- 📊 **Detailed Results**: Shows prediction with confidence scores and probability breakdown
- 🔒 **Privacy-Focused**: Images are processed in real-time and not stored
- ⚡ **Fast Processing**: Instant analysis and results
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Navigate to the project directory**:
   ```bash
   cd c:\Users\PC\Desktop\project_dl
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install Flask torch torchvision Pillow
   ```

## Running the Application

1. **Ensure the model file is in the project directory**:
   - The file `best_regnet_binary_model.pth` should be in the root directory

2. **Start the Flask server**:
   ```bash
   python app.py
   ```

3. **Open your browser and navigate to**:
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload an Image**:
   - Click "Browse Files" or drag and drop an image
   - Supported formats: PNG, JPEG, JPG
   - Maximum file size: 16MB

2. **Analyze**:
   - Click "Analyze Image" button
   - Wait for the AI to process the image

3. **View Results**:
   - See the prediction (Benign or Malignant)
   - Check the confidence score
   - Review detailed probability breakdown

4. **New Analysis**:
   - Click "New Analysis" to analyze another image

## Project Structure

```
project_dl/
│
├── app.py                          # Flask application
├── best_regnet_binary_model.pth    # Trained model weights
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── templates/
│   ├── index.html                  # Main page
│   └── about.html                  # About page
│
└── static/
    ├── css/
    │   └── style.css               # Styles
    └── js/
        └── script.js               # JavaScript functionality
```

## Model Information

- **Architecture**: RegNet-Y 400MF
- **Task**: Binary Classification (Benign vs Malignant)
- **Input Size**: 224x224 RGB images
- **Framework**: PyTorch
- **Output**: 2 classes with confidence scores

## Important Notes

⚠️ **Medical Disclaimer**: This application is designed for educational and research purposes only. It should NOT be used as a replacement for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Troubleshooting

### Model Loading Issues

If you encounter errors loading the model, ensure:
- The model file `best_regnet_binary_model.pth` is in the correct location
- The model was trained with the same RegNet architecture
- PyTorch is properly installed

### Port Already in Use

If port 5000 is already in use, modify the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to different port
```

### CUDA/GPU Issues

The application automatically uses GPU if available, otherwise falls back to CPU. For CPU-only:
```python
device = torch.device('cpu')
```

## Customization

### Change Model Architecture

If your model uses a different architecture, modify the `load_model()` function in `app.py`.

### Adjust Image Preprocessing

Modify the `transform` pipeline in `app.py` to match your model's training preprocessing.

### Update UI Colors

Edit the CSS variables in `static/css/style.css`:
```css
:root {
    --primary-color: #6366f1;
    --secondary-color: #8b5cf6;
    /* ... */
}
```

## License

This project is for educational purposes. Please ensure compliance with relevant medical data regulations and privacy laws when using in production.

## Support

For issues or questions, please check:
- Model file is correctly placed
- All dependencies are installed
- Python version compatibility
- Network/firewall settings for local server

---

Made with ❤️ for medical AI research
# breast-cancer-detection
