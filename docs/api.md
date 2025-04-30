# API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API does not require authentication for development purposes.

## Endpoints

### 1. Resume Management

#### Upload Resume
```http
POST /upload-resume
```

**Request**
- Content-Type: multipart/form-data
- Body:
  - file: PDF or DOCX file (max 10MB)

**Response**
```json
{
    "resume_id": "1745997736.056202",
    "matches": [
        {
            "job_id": "jd_1",
            "score": 0.85,
            "title": "Data Scientist",
            "company": "Data Analytics Inc",
            "description": "...",
            "posting_date": "2024-03-20T10:00:00Z"
        }
    ]
}
```

**Status Codes**
- 200: Success
- 400: Invalid file format or size
- 500: Server error

#### Get Resume Details
```http
GET /resume/{resume_id}
```

**Response**
```json
{
    "content": "Resume text content..."
}
```

**Status Codes**
- 200: Success
- 404: Resume not found

### 2. Job Management

#### Scrape LinkedIn Jobs
```http
POST /scrape-linkedin-jobs
```

**Request**
```json
{
    "keywords": "python developer",
    "location": "India",
    "num_pages": 1
}
```

**Response**
```json
{
    "message": "Successfully scraped 10 jobs",
    "jobs": [
        {
            "id": "jd_1",
            "title": "Python Developer",
            "company": "Tech Corp",
            "description": "...",
            "posting_date": "2024-03-20T10:00:00Z"
        }
    ]
}
```

**Status Codes**
- 200: Success
- 500: Server error

#### Get Job Details
```http
GET /job/{job_id}
```

**Response**
```json
{
    "id": "jd_1",
    "title": "Python Developer",
    "company": "Tech Corp",
    "description": "...",
    "posting_date": "2024-03-20T10:00:00Z"
}
```

**Status Codes**
- 200: Success
- 404: Job not found

### 3. Matching

#### Get Matches
```http
GET /matches/{resume_id}
```

**Response**
```json
{
    "matches": [
        {
            "job_id": "jd_1",
            "score": 0.85,
            "title": "Python Developer",
            "company": "Tech Corp"
        }
    ]
}
```

**Status Codes**
- 200: Success
- 404: Resume not found

### 4. Monitoring

#### Get Metrics
```http
GET /metrics
```

**Response**
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/upload-resume",status="success"} 10

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="POST",endpoint="/upload-resume",le="0.1"} 5
```

**Status Codes**
- 200: Success

## Error Responses

### Validation Error
```json
{
    "detail": "File too large. Maximum size allowed is 10MB"
}
```

### Not Found Error
```json
{
    "detail": "Resume not found"
}
```

### Server Error
```json
{
    "detail": "An unexpected error occurred"
}
```

## Rate Limiting
- 100 requests per minute per IP
- 1000 requests per hour per IP

## CORS
- All origins allowed for development
- Methods: GET, POST, PUT, DELETE
- Headers: Content-Type, Authorization

## File Upload Limits
- Maximum file size: 10MB
- Allowed formats: PDF, DOCX
- Maximum concurrent uploads: 5

## Response Times
- Average: < 500ms
- 95th percentile: < 1s
- 99th percentile: < 2s

## Monitoring Metrics

### Request Metrics
- `http_requests_total`: Total number of HTTP requests
- `http_request_duration_seconds`: Request latency
- `resume_processing_seconds`: Resume processing time
- `jd_scraping_seconds`: Job scraping time

### Processing Metrics
- `resume_processing_success_total`: Successful resume processing
- `resume_processing_failure_total`: Failed resume processing
- `job_scraping_success_total`: Successful job scraping
- `job_scraping_failure_total`: Failed job scraping

### Matching Metrics
- `matching_requests_total`: Total matching requests
- `matching_duration_seconds`: Matching process duration
- `match_quality_score`: Quality score of matches 