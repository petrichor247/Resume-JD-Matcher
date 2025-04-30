const API_BASE_URL = 'http://localhost:8000';

export const api = {
    // Upload and process resume
    uploadResume: async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/upload-resume`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to upload resume');
        }
        
        return response.json();
    },

    // Get job matches for a resume
    getMatches: async (resumeId) => {
        const response = await fetch(`${API_BASE_URL}/matches/${resumeId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get matches');
        }
        
        return response.json();
    },
}; 