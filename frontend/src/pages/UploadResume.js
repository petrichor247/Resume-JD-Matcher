import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadResume } from '../services/api';
import '../styles/UploadResume.css';

const UploadResume = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid PDF file');
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('resume', file);
      
      const response = await uploadResume(formData);
      if (response.success) {
        navigate('/matches');
      } else {
        setError(response.message || 'Failed to upload resume');
      }
    } catch (err) {
      setError('An error occurred while uploading the resume');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-header">
        <h1>Upload Your Resume</h1>
        <p>Upload your resume in PDF format to find matching job opportunities</p>
      </div>

      <div className="upload-section">
        <div className="upload-area">
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            className="file-input"
            id="resume-upload"
          />
          <label htmlFor="resume-upload" className="file-label">
            {file ? file.name : 'Choose PDF file'}
          </label>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button
          className="upload-button"
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading ? 'Uploading...' : 'Upload and Go'}
        </button>
      </div>
    </div>
  );
};

export default UploadResume; 