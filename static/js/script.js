// Get DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadForm = document.getElementById('uploadForm');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

let selectedFile = null;

// Drag and drop events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change event
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Click on upload area to trigger file input
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Handle file selection
function handleFile(file) {
    // Validate file type
    // Note: Both .jpg and .jpeg files use MIME type 'image/jpeg'
    const validTypes = ['image/png', 'image/jpeg'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, JPEG)');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

// Display image preview
function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    
    reader.readAsDataURL(file);
}

// Remove image
removeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    resetUpload();
});

// Reset upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    // Show loading spinner
    previewSection.style.display = 'none';
    loadingSpinner.style.display = 'block';
    resultsSection.style.display = 'none';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // Send request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            alert(data.error || 'An error occurred during prediction');
            previewSection.style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to the server. Please try again.');
        previewSection.style.display = 'block';
    } finally {
        loadingSpinner.style.display = 'none';
    }
});

// Display results
function displayResults(data) {
    // Set result image
    document.getElementById('resultImage').src = 'data:image/jpeg;base64,' + data.image;

    // Set prediction
    const predictionCard = document.getElementById('predictionCard');
    const predictionResult = document.getElementById('predictionResult');
    predictionResult.textContent = data.prediction;

    // Update card color based on prediction
    if (data.prediction.toLowerCase() === 'benign') {
        predictionCard.classList.remove('malignant');
        predictionCard.classList.add('benign');
    } else {
        predictionCard.classList.remove('benign');
        predictionCard.classList.add('malignant');
    }

    // Set confidence
    document.getElementById('confidenceValue').textContent = data.confidence + '%';
    const progressFill = document.getElementById('progressFill');
    progressFill.style.width = data.confidence + '%';

    // Set probabilities
    document.getElementById('benignProb').textContent = data.probabilities.Benign + '%';
    document.getElementById('malignantProb').textContent = data.probabilities.Malignant + '%';

    // Show results section
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// New analysis button
newAnalysisBtn.addEventListener('click', () => {
    resetUpload();
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Prevent default drag behavior on document
document.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
});
