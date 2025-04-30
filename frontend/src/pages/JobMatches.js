import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getJobMatches } from '../services/api';
import '../styles/JobMatches.css';

const JobMatches = () => {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchMatches = async () => {
      try {
        const response = await getJobMatches();
        if (response.success) {
          setMatches(response.matches);
        } else {
          setError(response.message || 'Failed to fetch job matches');
        }
      } catch (err) {
        setError('An error occurred while fetching job matches');
      } finally {
        setLoading(false);
      }
    };

    fetchMatches();
  }, []);

  const handleBack = () => {
    navigate('/upload');
  };

  if (loading) {
    return (
      <div className="matches-container">
        <div className="loading-container">
          <p>Finding the best matches for your resume...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="matches-container">
        <div className="error-container">
          <p>{error}</p>
          <button className="try-again-button" onClick={handleBack}>
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (matches.length === 0) {
    return (
      <div className="matches-container">
        <div className="no-matches">
          <p>No matching jobs found for your resume.</p>
          <button className="upload-another-button" onClick={handleBack}>
            Upload Another Resume
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="matches-container">
      <div className="matches-header">
        <h1>Job Matches</h1>
        <button className="back-button" onClick={handleBack}>
          Back to Upload
        </button>
      </div>

      <div className="matches-list">
        {matches.map((match, index) => (
          <div key={match.id} className="match-card">
            <div className="match-header">
              <h2>{match.title}</h2>
              <div className={`match-score ${getScoreClass(match.score)}`}>
                <span className="score-value">{match.score}%</span>
                <span className="score-label">Match</span>
              </div>
            </div>

            <div className="match-details">
              <p><strong>Company:</strong> {match.company}</p>
              <p><strong>Location:</strong> {match.location}</p>
              <p><strong>Job Type:</strong> {match.job_type}</p>
            </div>

            <div className="match-description">
              <h3>Description</h3>
              <p>{match.description}</p>
            </div>

            <div className="match-requirements">
              <h3>Requirements</h3>
              <ul>
                {match.requirements.map((req, idx) => (
                  <li key={idx}>{req}</li>
                ))}
              </ul>
            </div>

            <div className="match-actions">
              <a
                href={match.apply_url}
                target="_blank"
                rel="noopener noreferrer"
                className="apply-button"
              >
                Apply Now
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const getScoreClass = (score) => {
  if (score >= 80) return 'high-score';
  if (score >= 60) return 'medium-score';
  return 'low-score';
};

export default JobMatches; 