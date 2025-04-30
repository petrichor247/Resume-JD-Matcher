/**
 * API service for interacting with the backend
 */

const API_BASE_URL = 'http://localhost:5000/api';

/**
 * Upload a resume file
 * @param {File} file - The resume file to upload
 * @returns {Promise} - Promise with the upload response
 */
export const uploadResume = async (file) => {
  const formData = new FormData();
  formData.append('resume', file);

  try {
    const response = await fetch(`${API_BASE_URL}/resumes/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading resume:', error);
    throw error;
  }
};

/**
 * Get job matches for a resume
 * @param {string} resumeId - The ID of the resume
 * @returns {Promise} - Promise with the job matches
 */
export const getJobMatches = async (resumeId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/matches/${resumeId}`);

    if (!response.ok) {
      throw new Error(`Failed to get matches: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting job matches:', error);
    throw error;
  }
};

/**
 * Get all available job descriptions
 * @returns {Promise} - Promise with the job descriptions
 */
export const getJobDescriptions = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/jobs`);

    if (!response.ok) {
      throw new Error(`Failed to get jobs: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting job descriptions:', error);
    throw error;
  }
}; 