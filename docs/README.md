# Resume-Job Matching System Documentation

## User Manual

### Overview
The Resume-Job Matching System is an intelligent platform that helps job seekers find relevant job opportunities by matching their resumes with available job descriptions. The system uses the Siamese Neural Network model to analyze both resumes and job descriptions to find the best matches.

### Getting Started

#### Prerequisites
- A modern web browser (Chrome, Firefox, Safari, or Edge)
- PDF or DOCX format resume
- Internet connection

#### Accessing the Application
1. Open your web browser
2. Navigate to `http://localhost:3000`
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
   - HTML/CSS/JS
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
   - File directories based data storage
   - DVC for tracking data changes

4. **ML Layer**
   - Model training
   - Inference
   - Continuous learning
   - MLflow for tracking

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
   - Effective for text similarity
   - Can learn from pairs - Good for matching tasks
   - Continuous improvement

## Module Responsibilities

### `data_processor.py`
- Saves and retrieves resume/job data.
- Converts resume text to a storable format.
- Checks if model retraining is needed based on new resumes.

### `model_trainer.py`
- Loads and trains a Siamese similarity model.
- Computes embeddings for resumes and jobs.
- Finds job matches based on similarity scores.

### `data_downloader.py`
- Processes all existing resume files in batch.
- Useful for initial resume ingestion.

### `preprocess_jds.py`
- Extracts and normalizes job descriptions.
- Prepares job data for similarity scoring.

### `preprocessing/resumes/preprocess.py`
- Extracts and cleans resume content.
- Handles PDFs and DOCX formats.

### `preprocessing/jds/scraper.py`
- Scrapes job descriptions (e.g., from LinkedIn).

### `similarity/model.py`
- Wraps similarity logic using embeddings.

  
## Low-Level Design
Backend running on PORT 8000
Frontend running on PORT 3000
### API Endpoints



#### `GET /`
Serves the homepage.

#### `POST /upload-resume`
Accepts `.pdf` or `.docx` resumes. Extracts text, stores content, triggers matching, and schedules retraining.

#### `POST /add-job`
Takes a JSON job description and stores it.

#### `GET /job/{job_id}`
Returns metadata and description of a job.

#### `GET /resume/{resume_id}`
Returns metadata and matched jobs for a resume.

#### `GET /upload`, `GET /matches`
Serve frontend pages.

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

## Miscellaneous 


1. Feedback driven **model retraining**
2. Scraping **scheduled** through cron
3. **Logs** are saved to `data/logs/api.log` and printed to stdout.
5. **Error Handling** and detailed messages provided via `HTTPException`.
