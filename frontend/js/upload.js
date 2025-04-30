// Import API module
import { api } from './api.js';

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('resume-file');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadStatus = document.getElementById('upload-status');
    const uploadResult = document.getElementById('upload-result');
    const uploadError = document.getElementById('upload-error');
    const errorMessage = document.getElementById('error-message');
    
    // Update file name display when a file is selected
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
        } else {
            fileNameDisplay.textContent = 'No file chosen';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Validate file input
        if (fileInput.files.length === 0) {
            showError('Please select a file to upload');
            return;
        }
        
        const file = fileInput.files[0];
        
        // Validate file type
        if (!file.name.endsWith('.pdf') && !file.name.endsWith('.docx')) {
            showError('Please upload a PDF or DOCX file');
            return;
        }
        
        // Show loading status
        showLoading();
        
        try {
            // Use the API module to upload the resume
            const data = await api.uploadResume(file);
            
            // Store resume ID in localStorage for later use
            localStorage.setItem('currentResumeId', data.resume_id);
            
            // Show success message
            showSuccess();
            
            // Redirect to matches page after a short delay
            setTimeout(() => {
                window.location.href = 'matches.html';
            }, 2000);
        } catch (error) {
            showError(error.message);
        }
    });
    
    // Helper functions
    function showLoading() {
        uploadForm.style.display = 'none';
        uploadStatus.classList.remove('hidden');
        uploadResult.classList.add('hidden');
        uploadError.classList.add('hidden');
    }
    
    function showSuccess() {
        uploadForm.style.display = 'none';
        uploadStatus.classList.add('hidden');
        uploadResult.classList.remove('hidden');
        uploadError.classList.add('hidden');
    }
    
    function showError(message) {
        uploadForm.style.display = 'block';
        uploadStatus.classList.add('hidden');
        uploadResult.classList.add('hidden');
        uploadError.classList.remove('hidden');
        errorMessage.textContent = message;
    }
}); 