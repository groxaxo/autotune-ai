// Autotune-AI Frontend JavaScript

let currentJobId = null;
let statusCheckInterval = null;

// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const uploadSection = document.getElementById('uploadSection');
const progressSection = document.getElementById('progressSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

const audioFileInput = document.getElementById('audioFile');
const vocalFileInput = document.getElementById('vocalFile');
const backingFileInput = document.getElementById('backingFile');

const singleFileUpload = document.getElementById('singleFileUpload');
const separatedFilesUpload = document.getElementById('separatedFilesUpload');

const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const processingInfo = document.getElementById('processingInfo');

const downloadBtn = document.getElementById('downloadBtn');
const newProcessBtn = document.getElementById('newProcessBtn');
const retryBtn = document.getElementById('retryBtn');
const errorMessage = document.getElementById('errorMessage');

const vibratoPreserveInput = document.getElementById('vibratoPreserve');
const rangeValue = document.querySelector('.range-value');

// Input mode toggle
document.querySelectorAll('input[name="inputMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.value === 'single') {
            singleFileUpload.style.display = 'block';
            separatedFilesUpload.style.display = 'none';
        } else {
            singleFileUpload.style.display = 'none';
            separatedFilesUpload.style.display = 'block';
        }
    });
});

// File input handlers
function updateFileLabel(input, label) {
    if (input.files && input.files[0]) {
        const fileName = input.files[0].name;
        const fileSize = (input.files[0].size / (1024 * 1024)).toFixed(2);
        label.querySelector('.upload-text').textContent = fileName;
        label.querySelector('.upload-hint').textContent = `${fileSize} MB`;
        label.classList.add('has-file');
    }
}

audioFileInput.addEventListener('change', function() {
    updateFileLabel(this, this.nextElementSibling);
});

vocalFileInput.addEventListener('change', function() {
    updateFileLabel(this, this.nextElementSibling);
});

backingFileInput.addEventListener('change', function() {
    updateFileLabel(this, this.nextElementSibling);
});

// Vibrato preserve slider
vibratoPreserveInput.addEventListener('input', function() {
    const value = (parseFloat(this.value) * 100).toFixed(0);
    rangeValue.textContent = `${value}%`;
});

// Form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    const inputMode = document.querySelector('input[name="inputMode"]:checked').value;
    
    // Validate file inputs
    if (inputMode === 'single') {
        if (!audioFileInput.files || audioFileInput.files.length === 0) {
            alert('Please select an audio file');
            return;
        }
        formData.append('audioFile', audioFileInput.files[0]);
    } else {
        if (!vocalFileInput.files || vocalFileInput.files.length === 0 ||
            !backingFileInput.files || backingFileInput.files.length === 0) {
            alert('Please select both vocal and backing files');
            return;
        }
        formData.append('vocalFile', vocalFileInput.files[0]);
        formData.append('backingFile', backingFileInput.files[0]);
        formData.append('audioFile', new File([], ''));  // Empty file to satisfy backend
    }
    
    // Add all settings to form data
    formData.append('mode', document.getElementById('mode').value);
    formData.append('rootNote', document.getElementById('rootNote').value);
    formData.append('scale', document.getElementById('scale').value);
    formData.append('vibratoPreserve', document.getElementById('vibratoPreserve').value);
    formData.append('separationModel', document.getElementById('separationModel').value);
    formData.append('pitchMethod', document.getElementById('pitchMethod').value);
    formData.append('vocoderMethod', document.getElementById('vocoderMethod').value);
    formData.append('sampleRate', document.getElementById('sampleRate').value);
    formData.append('denoise', document.getElementById('denoise').checked);
    formData.append('vad', document.getElementById('vad').checked);
    formData.append('align', document.getElementById('align').checked);
    
    try {
        // Show progress section
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        errorSection.style.display = 'none';
        
        progressFill.style.width = '0%';
        progressText.textContent = 'Uploading files...';
        
        // Upload files and start processing
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        currentJobId = data.job_id;
        progressText.textContent = 'Processing started...';
        
        // Start checking status
        startStatusChecking();
        
    } catch (error) {
        console.error('Upload error:', error);
        showError(error.message);
    }
});

// Status checking
function startStatusChecking() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    statusCheckInterval = setInterval(checkStatus, 2000);
    checkStatus(); // Check immediately
}

async function checkStatus() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`/status/${currentJobId}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Status check failed');
        }
        
        updateProgress(data);
        
        if (data.status === 'completed') {
            clearInterval(statusCheckInterval);
            showResult(data);
        } else if (data.status === 'failed') {
            clearInterval(statusCheckInterval);
            showError(data.error || 'Processing failed');
        }
        
    } catch (error) {
        console.error('Status check error:', error);
        // Continue checking, don't show error immediately
    }
}

function updateProgress(data) {
    const progress = data.progress || 0;
    progressFill.style.width = `${progress}%`;
    
    let statusText = '';
    switch (data.status) {
        case 'queued':
            statusText = 'Queued for processing...';
            break;
        case 'processing':
            statusText = getProcessingStageText(progress);
            break;
        case 'completed':
            statusText = 'Processing complete!';
            break;
        case 'failed':
            statusText = 'Processing failed';
            break;
        default:
            statusText = 'Processing...';
    }
    
    progressText.textContent = statusText;
    
    // Update processing info
    if (data.params) {
        updateProcessingInfo(data.params);
    }
}

function getProcessingStageText(progress) {
    if (progress < 20) return 'Preparing files...';
    if (progress < 40) return 'Separating vocals...';
    if (progress < 60) return 'Extracting pitch...';
    if (progress < 80) return 'Correcting pitch...';
    if (progress < 95) return 'Mixing final audio...';
    return 'Finalizing...';
}

function updateProcessingInfo(params) {
    const notes = {
        57: 'A3', 58: 'A#3', 59: 'B3', 60: 'C4', 61: 'C#4', 62: 'D4',
        63: 'D#4', 64: 'E4', 65: 'F4', 66: 'F#4', 67: 'G4', 68: 'G#4', 69: 'A4'
    };
    
    processingInfo.innerHTML = `
        <p><strong>Settings:</strong></p>
        <p>Mode: ${params.mode}</p>
        <p>Key: ${notes[params.root_midi] || params.root_midi} ${params.scale}</p>
        <p>Vibrato: ${(params.vibrato_preserve * 100).toFixed(0)}%</p>
        <p>Pitch Method: ${params.pitch_method}</p>
        <p>Vocoder: ${params.vocoder_method}</p>
    `;
}

function showResult(data) {
    progressSection.style.display = 'none';
    resultSection.style.display = 'block';
    
    // Set download button handler
    downloadBtn.onclick = () => {
        window.location.href = data.download_url;
    };
}

function showError(message) {
    progressSection.style.display = 'none';
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
}

// New process button
newProcessBtn.addEventListener('click', () => {
    resetForm();
});

retryBtn.addEventListener('click', () => {
    resetForm();
});

function resetForm() {
    currentJobId = null;
    
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    
    uploadSection.style.display = 'block';
    progressSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Reset form
    uploadForm.reset();
    
    // Reset file labels
    document.querySelectorAll('.file-label').forEach(label => {
        label.classList.remove('has-file');
        const textSpan = label.querySelector('.upload-text');
        const hintSpan = label.querySelector('.upload-hint');
        
        if (label.getAttribute('for') === 'audioFile') {
            textSpan.textContent = 'Click to select audio file';
            hintSpan.textContent = 'WAV, MP3, FLAC supported (max 500MB)';
        } else if (label.getAttribute('for') === 'vocalFile') {
            textSpan.textContent = 'Vocal Track';
            hintSpan.textContent = '';
        } else if (label.getAttribute('for') === 'backingFile') {
            textSpan.textContent = 'Backing Track';
            hintSpan.textContent = '';
        }
    });
    
    // Reset vibrato slider display
    rangeValue.textContent = '25%';
}

// Health check on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('Service health:', data);
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
