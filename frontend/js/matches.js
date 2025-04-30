// Import API module
import { api } from './api.js';

document.addEventListener('DOMContentLoaded', function() {
    const resumeSelect = document.getElementById('resume-select');
    const findMatchesButton = document.getElementById('find-matches');
    const loadingContainer = document.getElementById('loading');
    const noResumeContainer = document.getElementById('no-resume');
    const noMatchesContainer = document.getElementById('no-matches');
    const matchesContainer = document.getElementById('matches-container');
    const matchesList = document.getElementById('matches-list');
    const matchesCount = document.getElementById('matches-count');
    
    // Load resumes from localStorage
    loadResumes();
    
    // Check if there's a current resume ID in localStorage
    const currentResumeId = localStorage.getItem('currentResumeId');
    if (currentResumeId) {
        // Add the current resume to the select if it's not already there
        if (!Array.from(resumeSelect.options).some(option => option.value === currentResumeId)) {
            const option = document.createElement('option');
            option.value = currentResumeId;
            option.textContent = `Resume ${currentResumeId.substring(0, 8)}...`;
            resumeSelect.appendChild(option);
        }
        
        // Select the current resume
        resumeSelect.value = currentResumeId;
        
        // Find matches for the current resume
        findMatches(currentResumeId);
    }
    
    // Handle find matches button click
    findMatchesButton.addEventListener('click', function() {
        const selectedResumeId = resumeSelect.value;
        if (selectedResumeId) {
            findMatches(selectedResumeId);
        } else {
            showNoResumeMessage();
        }
    });
    
    // Function to load resumes
    function loadResumes() {
        // In a real application, you would fetch resumes from the server
        // For now, we'll just use the current resume ID from localStorage
        const currentResumeId = localStorage.getItem('currentResumeId');
        
        if (currentResumeId) {
            const option = document.createElement('option');
            option.value = currentResumeId;
            option.textContent = `Resume ${currentResumeId.substring(0, 8)}...`;
            resumeSelect.appendChild(option);
        }
    }
    
    // Function to find matches for a resume
    async function findMatches(resumeId) {
        // Show loading
        showLoading();
        
        try {
            // Use the API module to get matches
            const data = await api.getMatches(resumeId);
            
            // Hide loading
            hideLoading();
            
            // Display matches
            displayMatches(data.matches);
        } catch (error) {
            // Hide loading
            hideLoading();
            
            // Show error
            showError(error.message);
        }
    }
    
    // Function to display matches
    function displayMatches(matches) {
        // Clear previous matches
        matchesList.innerHTML = '';
        
        // Update matches count
        matchesCount.textContent = matches.length;
        
        if (matches.length === 0) {
            // Show no matches message
            showNoMatchesMessage();
        } else {
            // Hide no matches message
            hideNoMatchesMessage();
            
            // Show matches container
            matchesContainer.classList.remove('hidden');
            
            // Add match cards
            matches.forEach(match => {
                const matchCard = createMatchCard(match);
                matchesList.appendChild(matchCard);
            });
        }
    }
    
    // Function to create a match card
    function createMatchCard(match) {
        const card = document.createElement('div');
        card.className = 'match-card';
        
        // Format date
        const postingDate = match.posting_date ? new Date(match.posting_date).toLocaleDateString() : 'N/A';
        
        // Format score as percentage
        const scorePercentage = Math.round(match.score * 100);
        
        card.innerHTML = `
            <div class="match-header">
                <div class="match-title">${match.title}</div>
                <div class="match-score">${scorePercentage}% Match</div>
            </div>
            <div class="match-company">${match.company}</div>
            <div class="match-description">${truncateText(match.description, 200)}</div>
            <div class="match-date">Posted: ${postingDate}</div>
        `;
        
        return card;
    }
    
    // Helper function to truncate text
    function truncateText(text, maxLength) {
        if (text.length <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength) + '...';
    }
    
    // Helper functions for UI state
    function showLoading() {
        loadingContainer.classList.remove('hidden');
        noResumeContainer.classList.add('hidden');
        noMatchesContainer.classList.add('hidden');
        matchesContainer.classList.add('hidden');
    }
    
    function hideLoading() {
        loadingContainer.classList.add('hidden');
    }
    
    function showNoResumeMessage() {
        noResumeContainer.classList.remove('hidden');
        noMatchesContainer.classList.add('hidden');
        matchesContainer.classList.add('hidden');
    }
    
    function showNoMatchesMessage() {
        noMatchesContainer.classList.remove('hidden');
        matchesContainer.classList.add('hidden');
    }
    
    function hideNoMatchesMessage() {
        noMatchesContainer.classList.add('hidden');
    }
    
    function showError(message) {
        // Create error element if it doesn't exist
        let errorContainer = document.getElementById('error-container');
        if (!errorContainer) {
            errorContainer = document.createElement('div');
            errorContainer.id = 'error-container';
            errorContainer.className = 'error-container';
            document.querySelector('.matches-section').appendChild(errorContainer);
        }
        
        // Set error message
        errorContainer.innerHTML = `<p>${message}</p>`;
        errorContainer.classList.remove('hidden');
        
        // Hide other containers
        noResumeContainer.classList.add('hidden');
        noMatchesContainer.classList.add('hidden');
        matchesContainer.classList.add('hidden');
    }
}); 