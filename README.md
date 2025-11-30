# 🩺 Breast Cancer Detection AI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg) ![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg) ![License](https://img.shields.io/badge/License-Educational-yellow.svg)

A Flask web application that uses deep learning (RegNet model) to detect breast cancer from medical images. The application provides a modern, user-friendly interface for uploading images and receiving instant AI-powered predictions.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Dataset](#dataset) • [Model Info](#model-information)

---

## ✨ Features

- 🧠 **Advanced AI Model**: Uses RegNet-Y 400MF architecture for accurate predictions
- 🎨 **Modern UI**: Beautiful, responsive design with gradient backgrounds
- 📊 **Detailed Results**: Shows prediction with confidence scores and probability breakdown
- 🔒 **Privacy-Focused**: Images are processed in real-time and not stored
- ⚡ **Fast Processing**: Instant analysis and results
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- 🎯 **High Accuracy**: Trained on comprehensive breast cancer histopathology dataset

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Troubleshooting](#troubleshooting)
- [Customization](#customization)
- [Contributors](#contributors)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: Version 3.8 or higher
- **pip**: Python package manager
- **Git** (optional): For cloning the repository

---

## 📥 Installation

### Step 1: Clone or Download the Project

```bash
# Clone the repository (if using Git)
git clone <repository-url>

# Navigate to the project directory
cd project_dl
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option 1: Using requirements.txt**
```bash
pip install -r requirements.txt
```

**Option 2: Manual Installation**
```bash
pip install Flask torch torchvision Pillow
```

### Step 4: Verify Model File

Ensure the trained model file is in the project root:

```
project_dl/
├── best_regnet_binary_model.pth  ← This file should be here
├── app.py
└── ...
```

---

## 🚀 Running the Application

1. **Activate your virtual environment** (if not already activated)

2. **Start the Flask server:**
   ```bash
   python app.py
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

4. You should see the application homepage!

---

## 💡 Usage

### Step-by-Step Guide

1. **Upload an Image**
   - Click the "Browse Files" button or drag and drop an image
   - Supported formats: PNG, JPEG, JPG
   - Maximum file size: 16MB

2. **Analyze**
   - Click the "Analyze Image" button
   - The AI will process the image in real-time

3. **View Results**
   - **Prediction**: Benign or Malignant classification
   - **Confidence Score**: Overall model confidence
   - **Probability Breakdown**: Detailed percentages for each class

4. **New Analysis**
   - Click "New Analysis" to upload and analyze another image

---

## 📊 Dataset

This model was trained on the **5C Breast Cancer Dataset**, sourced from Roboflow and imported to Kaggle for this project.

### Dataset Links

- 🔗 **Kaggle Dataset**: [Breast Cancer Dataset ](https://www.kaggle.com/datasets/aminebahyoul/breast-cancer-dataset)
- 🔗 **Kaggle Notebook**: [Project DL Training Notebook](https://www.kaggle.com/code/meeeeeeeeeeeeeed/projet-dl)
- 🔗 **Original Source**: [5C_Breast Cancer on Roboflow Universe](https://universe.roboflow.com/med-qhtrw/5c_breast-cancer-cgbl1)

### Dataset Details

| Property | Description |
|----------|-------------|
| **Name** | 5C_Breast Cancer > projet_dl |
| **Original Source** | Roboflow Universe (med-qhtrw) |
| **Kaggle Import** | By Amine Bahyoul |
| **License** | Public Domain |
| **Type** | Medical imaging dataset |
| **Classes** | Binary classification (Benign/Malignant) |
| **Format** | High-resolution medical images |
| **Purpose** | Training deep learning models for breast cancer detection |

### Dataset Pipeline

1. **Source**: Roboflow Universe (med-qhtrw/5c_breast-cancer-cgbl1)
2. **Processing**: Dataset preparation and organization
3. **Distribution**: Imported to Kaggle for accessibility
4. **Training**: Used for RegNet model training (see Kaggle notebook)

### Citation

If you use this dataset or model, please cite:

```
5C_Breast Cancer Dataset. Roboflow Universe.
https://universe.roboflow.com/med-qhtrw/5c_breast-cancer-cgbl1

Breast Cancer Dataset. Kaggle.
https://www.kaggle.com/datasets/meeeeeeeeeeeeeed/breast-cancer

Project Training Notebook. Kaggle.
https://www.kaggle.com/code/meeeeeeeeeeeeeed/projet-dl
```

---

## 📁 Project Structure

```
project_dl/
│
├── Evaluation                      # Evaluation model pictures
├── app.py                          # Flask application (main backend)
├── best_regnet_binary_model.pth    # Trained RegNet model weights
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── templates/
│   ├── index.html                  # Main application page
│   └── about.html                  # About/information page
│
└── static/
    ├── css/
    │   └── style.css               # Application styles
    └── js/
        └── script.js               # Frontend JavaScript
```

---

## 🤖 Model Information

### Architecture Specifications

| Property | Value |
|----------|-------|
| **Architecture** | RegNet-Y 400MF |
| **Framework** | PyTorch |
| **Task** | Binary Classification |
| **Input Size** | 224×224 RGB images |
| **Output Classes** | 2 (Benign, Malignant) |
| **Activation** | Softmax |

### Model Pipeline

1. **Image Preprocessing**
   - Resize to 224×224
   - Normalize with ImageNet statistics
   - Convert to tensor

2. **Inference**
   - Forward pass through RegNet
   - Softmax activation
   - Confidence score calculation

3. **Post-processing**
   - Class prediction
   - Probability distribution
   - Result formatting

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors

**Problem**: `FileNotFoundError` or model loading fails

**Solution**:
- Verify `best_regnet_binary_model.pth` is in the root directory
- Check file permissions
- Ensure the model was trained with the same architecture

```bash
# Check if file exists
ls -la best_regnet_binary_model.pth  # Linux/Mac
dir best_regnet_binary_model.pth     # Windows
```

#### 2. Port Already in Use

**Problem**: `Address already in use` error

**Solution**: Change the port in `app.py`:

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001 or any free port
```

#### 3. CUDA/GPU Issues

**Problem**: GPU not detected or CUDA errors

**Solution**: The app automatically falls back to CPU. To force CPU mode:

```python
device = torch.device('cpu')
```

#### 4. Memory Errors

**Problem**: Out of memory during inference

**Solution**:
- Close other applications
- Process smaller images
- Use CPU instead of GPU for inference

#### 5. Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 👥 Contributors

This project was developed as an **academic project** by Information Systems Engineering Master's students:

<table>
  <tr>
    <td align="center">
      <strong>Yahia Ezzahri</strong><br>
      Information Systems Engineer (Master)<br>
      Model Development & Backend
    </td>
    <td align="center">
      <strong>Amine Bahyoul</strong><br>
      Information Systems Engineer (Master)<br>
      Dataset Curation & ML Engineering
    </td>
    <td align="center">
      <strong>Mohamed Ait Messskine</strong><br>
      Information Systems Engineer (Master)<br>
      Frontend Development & Testing
    </td>
  </tr>
</table>

### Project Context

- 🎓 **Program**: Master's in Information Systems Engineering
- 📚 **Type**: Academic Research Project
- 🎯 **Objective**: Apply deep learning techniques to medical image analysis
- 🔬 **Domain**: Healthcare AI and Computer Vision

### Contributions

- **Yahia Ezzahri**: Model architecture design, Flask backend development, deployment, UI/UX design
- **Amine Bahyoul**: Dataset sourcing from Roboflow, Kaggle import, data preprocessing, model training and validation
- **Mohamed Ait Messskine**: Frontend development, user interface implementation, testing, documentation

---

## ⚠️ Medical Disclaimer

> **IMPORTANT**: This application is designed for **educational and research purposes only**.
>
> - ❌ **DO NOT** use as a replacement for professional medical diagnosis
> - ❌ **DO NOT** make medical decisions based solely on this tool
> - ✅ **ALWAYS** consult qualified healthcare professionals
> - ✅ **USE** only as a supplementary research tool
>
> The developers assume no liability for any medical decisions made using this application.

---

## 📄 License

This project is released for **educational and research purposes**.

### Usage Rights

- ✅ Educational use
- ✅ Research purposes
- ✅ Non-commercial applications
- ❌ Commercial use without permission
- ❌ Medical practice without proper certification

### Compliance Requirements

When using this project, ensure compliance with:
- HIPAA (Health Insurance Portability and Accountability Act)
- GDPR (General Data Protection Regulation)
- Local medical data regulations
- Patient privacy laws

---

## 🤝 Support & Contributing

### Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review Python version compatibility (3.8+)
3. Verify all dependencies are installed
4. Check firewall/network settings

### Contributing

We welcome contributions! To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out to the contributors.

---

## 🙏 Acknowledgments

- **Roboflow Universe** (med-qhtrw) for providing the original 5C Breast Cancer dataset
- **Kaggle Community** for hosting and distribution platform
- **PyTorch Team** for the deep learning framework
- **Flask Community** for the web framework
- **RegNet Authors** (Radosavovic et al.) for the model architecture
- Our academic supervisors and mentors for guidance throughout this project

---

**Made with ❤️ for medical AI research**

⭐ Star this repository if you found it helpful!
