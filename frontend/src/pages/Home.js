import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { uploadResume, getJobDescriptions } from '../services/api';
import '../styles/Home.css';

const Home = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [resumeId, setResumeId] = useState(null);
  const [error, setError] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    // Fetch job descriptions when component mounts
    const fetchJobs = async () => {
      try {
        const jobData = await getJobDescriptions();
        setJobs(jobData);
      } catch (err) {
        console.error('Error fetching jobs:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchJobs();
  }, []);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError(null);
    } else {
      setFile(null);
      setError('Please select a valid PDF file');
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const response = await uploadResume(file);
      setResumeId(response.resume_id);
      setUploadSuccess(true);
    } catch (err) {
      setError('Failed to upload resume. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  const handleCheckMatches = () => {
    navigate('/upload');
  };

  return (
    <div className="home-container">
      <header className="home-header">
        <h1>Resume-JD Search Engine</h1>
        <p>
          Welcome to our intelligent job matching platform. Upload your resume and discover
          the best matching job opportunities based on your skills and experience.
          Our advanced AI model analyzes your resume and matches it with relevant job
          descriptions to help you find your perfect career match.
        </p>
        <button className="check-matches-button" onClick={handleCheckMatches}>
          Check Job Matches
        </button>
      </header>

      <section className="upload-section">
        <h2>Upload Your Resume</h2>
        <form onSubmit={handleUpload} className="upload-form">
          <div className="file-input-container">
            <input
              type="file"
              id="resume-file"
              accept=".pdf"
              onChange={handleFileChange}
              className="file-input"
            />
            <label htmlFor="resume-file" className="file-label">
              {file ? file.name : 'Choose PDF file'}
            </label>
          </div>
          
          {error && <p className="error-message">{error}</p>}
          
          <button 
            type="submit" 
            className="upload-button"
            disabled={!file || uploading}
          >
            {uploading ? 'Uploading...' : 'Upload Resume'}
          </button>
        </form>

        {uploadSuccess && resumeId && (
          <div className="success-message">
            <p>Resume uploaded successfully!</p>
            <Link to={`/matches/${resumeId}`} className="view-matches-button">
              View Job Matches
            </Link>
          </div>
        )}
      </section>

      <section className="jobs-section">
        <h2>Available Jobs</h2>
        {loading ? (
          <p>Loading jobs...</p>
        ) : (
          <div className="jobs-list">
            {jobs.length > 0 ? (
              jobs.map((job) => (
                <div key={job.job_id} className="job-card">
                  <h3>{job.title}</h3>
                  <p className="company">{job.company}</p>
                  <p className="location">{job.location}</p>
                  <p className="description">{job.description.substring(0, 150)}...</p>
                </div>
              ))
            ) : (
              <p>No jobs available</p>
            )}
          </div>
        )}
      </section>
    </div>
  );
};

export default Home; 