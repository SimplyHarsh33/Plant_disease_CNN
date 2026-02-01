/**
 * PlantDoc AI - Plant Disease Detection
 * Frontend JavaScript for image upload and prediction
 */

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingContainer = document.getElementById('loadingContainer');
const resultsSection = document.getElementById('resultsSection');

// State
let selectedFile = null;

// ==================== Event Listeners ====================

// Click to upload
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop events
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    } else {
        showError('Please upload an image file (JPG, PNG, WEBP, or GIF)');
    }
});

// Remove image
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// Analyze button
analyzeBtn.addEventListener('click', () => {
    if (selectedFile) {
        analyzeImage(selectedFile);
    }
});

// New analysis button
document.getElementById('newAnalysisBtn').addEventListener('click', () => {
    resetUpload();
    resultsSection.style.display = 'none';
    uploadZone.style.display = 'block';
    previewContainer.style.display = 'none';
});

// ==================== Functions ====================

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPG, PNG, WEBP, or GIF)');
        return;
    }
    
    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadZone.style.display = 'none';
        previewContainer.style.display = 'block';
        previewContainer.classList.add('fade-in');
    };
    reader.readAsDataURL(file);
}

/**
 * Reset upload state
 */
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewContainer.style.display = 'none';
    uploadZone.style.display = 'block';
    loadingContainer.style.display = 'none';
    
    // Clear any previous errors
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
}

/**
 * Show error message
 */
function showError(message) {
    // Remove existing error
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Create error element
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message fade-in';
    errorDiv.textContent = message;
    
    // Insert after upload zone or preview container
    const section = document.querySelector('.upload-section');
    section.appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

/**
 * Analyze the uploaded image
 */
async function analyzeImage(file) {
    // Show loading state
    previewContainer.style.display = 'none';
    loadingContainer.style.display = 'block';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        // Send request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            displayResults(data.predictions);
        } else {
            showError(data.error || 'Failed to analyze image. Please try again.');
            resetUpload();
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Connection error. Please make sure the server is running.');
        resetUpload();
    }
}

/**
 * Display prediction results
 */
function displayResults(predictions) {
    loadingContainer.style.display = 'none';
    resultsSection.style.display = 'block';
    
    if (!predictions || predictions.length === 0) {
        showError('No predictions returned. Please try a different image.');
        return;
    }
    
    const mainPrediction = predictions[0];
    const isHealthy = mainPrediction.disease.toLowerCase() === 'healthy';
    
    // Update main result card
    const resultCard = document.getElementById('mainResult');
    resultCard.className = `result-card glass-card ${isHealthy ? 'healthy' : 'diseased'}`;
    
    // Status icon
    const statusIcon = document.getElementById('statusIcon');
    statusIcon.textContent = isHealthy ? '✅' : '⚠️';
    
    // Disease name and plant
    document.getElementById('diseaseName').textContent = mainPrediction.disease;
    document.getElementById('plantName').textContent = `Plant: ${mainPrediction.plant}`;
    
    // Confidence
    document.getElementById('confidenceValue').textContent = `${mainPrediction.confidence}%`;
    
    // Update confidence badge color based on confidence level
    const confidenceBadge = document.getElementById('confidenceBadge');
    if (mainPrediction.confidence >= 80) {
        confidenceBadge.style.borderColor = 'rgba(16, 185, 129, 0.5)';
    } else if (mainPrediction.confidence >= 50) {
        confidenceBadge.style.borderColor = 'rgba(245, 158, 11, 0.5)';
    } else {
        confidenceBadge.style.borderColor = 'rgba(239, 68, 68, 0.5)';
    }
    
    // Description and symptoms
    document.getElementById('diseaseDescription').textContent = mainPrediction.description;
    document.getElementById('diseaseSymptoms').textContent = mainPrediction.symptoms;
    
    // Treatment list
    const treatmentList = document.getElementById('treatmentList');
    treatmentList.innerHTML = '';
    mainPrediction.treatment.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        treatmentList.appendChild(li);
    });
    
    // Prevention
    document.getElementById('diseasePrevention').textContent = mainPrediction.prevention;
    
    // Other predictions
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';
    
    const otherPredictions = predictions.slice(1);
    if (otherPredictions.length > 0) {
        document.getElementById('otherPredictions').style.display = 'block';
        
        otherPredictions.forEach(pred => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <span class="prediction-name">${pred.plant} - ${pred.disease}</span>
                <span class="prediction-confidence">${pred.confidence}%</span>
            `;
            predictionsList.appendChild(item);
        });
    } else {
        document.getElementById('otherPredictions').style.display = 'none';
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ==================== Initialize ====================

// Prevent default drag behaviors on document
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    document.body.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

console.log('🌿 PlantDoc AI initialized');
