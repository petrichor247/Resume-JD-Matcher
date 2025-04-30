# Resume-Job Matching System Documentation

## Table of Contents
1. [User Manual](#user-manual)
2. [Design Document](#design-document)
3. [Architecture Overview](#architecture-overview)
4. [High-Level Design](#high-level-design)
5. [Low-Level Design](#low-level-design)
6. [API Documentation](#api-documentation)
7. [Monitoring and Metrics](#monitoring-and-metrics)

## User Manual

### Overview
The Resume-Job Matching System is an intelligent platform that helps job seekers find relevant job opportunities by matching their resumes with available job descriptions. The system uses advanced machine learning to analyze both resumes and job descriptions to find the best matches.

### Getting Started

#### Prerequisites
- A modern web browser (Chrome, Firefox, Safari, or Edge)
- PDF or DOCX format resume
- Internet connection

#### Accessing the Application
1. Open your web browser
2. Navigate to `http://localhost:8000`
3. You'll see the home page with options to:
   - Upload your resume
   - View job matches
   - Browse available jobs

### Using the Application

#### Uploading Your Resume
1. Click on the "Upload Resume" button
2. Select your resume file (PDF or DOCX format)
3. Wait for the upload to complete
4. The system will automatically process your resume and find matching jobs

#### Viewing Job Matches
1. After uploading your resume, you'll be redirected to the matches page
2. Jobs are displayed in order of relevance
3. For each job, you can see:
   - Job title
   - Company name
   - Match score
   - Job description
   - Required skills

#### Scraping New Jobs
1. Navigate to the "Jobs" section
2. Click "Scrape New Jobs"
3. Enter keywords for job search
4. Select location (default: India)
5. Choose number of pages to scrape
6. Click "Start Scraping"

### Troubleshooting

#### Common Issues
1. **Upload Failed**
   - Ensure your file is in PDF or DOCX format
   - Check file size (max 10MB)
   - Verify internet connection

2. **No Matches Found**
   - Try uploading a more detailed resume
   - Check if jobs are available in the system
   - Try scraping new jobs

3. **Slow Performance**
   - Clear browser cache
   - Check internet connection
   - Try during off-peak hours

## Design Document

### System Architecture

```
[Frontend] <-> [Backend API] <-> [Data Processing] <-> [ML Model]
    ^              ^                  ^                  ^
    |              |                  |                  |
[User Interface] [FastAPI] [Data Processor] [Siamese Network]
```

### Components

1. **Frontend**
   - React-based web interface
   - Responsive design
   - Real-time updates

2. **Backend**
   - FastAPI server
   - RESTful API endpoints
   - Async processing

3. **Data Processing**
   - Resume parsing
   - Job description processing
   - Text preprocessing

4. **Machine Learning**
   - TF-IDF vectorization
   - Siamese network for matching
   - Continuous learning

### Data Flow

1. **Resume Processing**
   ```
   Upload -> Parse -> Preprocess -> Vectorize -> Store
   ```

2. **Job Processing**
   ```
   Scrape -> Parse -> Preprocess -> Vectorize -> Store
   ```

3. **Matching Process**
   ```
   Query -> Vectorize -> Compare -> Rank -> Return
   ```

## Architecture Overview

### System Components

1. **Web Interface**
   - Single Page Application
   - Real-time updates
   - Responsive design

2. **API Layer**
   - RESTful endpoints
   - Authentication
   - Rate limiting

3. **Processing Layer**
   - Text extraction
   - Feature engineering
   - Data storage

4. **ML Layer**
   - Model training
   - Inference
   - Continuous learning

5. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Performance tracking

### Data Storage

1. **Raw Data**
   - Resumes (PDF/DOCX)
   - Job descriptions
   - Metadata

2. **Processed Data**
   - Text embeddings
   - Feature vectors
   - Match scores

3. **Model Data**
   - Trained models
   - Vectorizers
   - Configuration

## High-Level Design

### Design Choices

1. **FastAPI Framework**
   - High performance
   - Async support
   - Automatic documentation
   - Type checking

2. **TF-IDF Vectorization**
   - Efficient text representation
   - Language independence
   - Easy to update

3. **Siamese Network**
   - Effective for matching
   - Can learn from new data
   - Handles varying lengths

4. **Docker Containerization**
   - Easy deployment
   - Consistent environment
   - Scalable architecture

### Rationale

1. **Why FastAPI?**
   - Modern async framework
   - Excellent performance
   - Built-in OpenAPI support
   - Easy to maintain

2. **Why TF-IDF?**
   - Proven technology
   - Fast processing
   - Good for text matching
   - Easy to understand

3. **Why Siamese Network?**
   - Effective for similarity
   - Can learn from pairs
   - Good for matching tasks
   - Continuous improvement

## Low-Level Design

### API Endpoints

1. **Resume Management**
   ```python
   POST /upload-resume
   - Input: PDF/DOCX file
   - Output: {resume_id, matches}
   - Status: 200 OK, 400 Bad Request, 500 Server Error
   ```

2. **Job Management**
   ```python
   POST /scrape-linkedin-jobs
   - Input: {keywords, location, num_pages}
   - Output: {message, jobs}
   - Status: 200 OK, 500 Server Error
   ```

3. **Matching**
   ```python
   GET /matches/{resume_id}
   - Input: resume_id
   - Output: {matches}
   - Status: 200 OK, 404 Not Found
   ```

### Data Structures

1. **Resume**
   ```python
   {
       'id': str,
       'content': str,
       'embedding': np.array,
       'metadata': {
           'upload_date': datetime,
           'file_type': str,
           'file_size': int
       }
   }
   ```

2. **Job**
   ```python
   {
       'id': str,
       'title': str,
       'company': str,
       'description': str,
       'embedding': np.array,
       'metadata': {
           'posting_date': datetime,
           'location': str,
           'source': str
       }
   }
   ```

3. **Match**
   ```python
   {
       'resume_id': str,
       'job_id': str,
       'score': float,
       'metadata': {
           'match_date': datetime,
           'confidence': float
       }
   }
   ```

## Monitoring and Metrics

### Prometheus Metrics

1. **Request Metrics**
   - `http_requests_total`
   - `http_request_duration_seconds`
   - `resume_processing_seconds`
   - `jd_scraping_seconds`

2. **Processing Metrics**
   - `resume_processing_success_total`
   - `resume_processing_failure_total`
   - `job_scraping_success_total`
   - `job_scraping_failure_total`

3. **Matching Metrics**
   - `matching_requests_total`
   - `matching_duration_seconds`
   - `match_quality_score`

### Grafana Dashboards

1. **System Overview**
   - Request rates
   - Error rates
   - Processing times

2. **Resume Processing**
   - Upload success rate
   - Processing time
   - Error distribution

3. **Job Matching**
   - Match quality
   - Processing time
   - Success rate

### Alerting

1. **Performance Alerts**
   - High latency
   - Error rate threshold
   - Processing time threshold

2. **System Alerts**
   - Service availability
   - Resource usage
   - Error patterns 