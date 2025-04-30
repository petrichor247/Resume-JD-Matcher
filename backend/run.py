from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import PyPDF2
import docx
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from data_downloader import DataDownloader
import preprocess_jds as  pjds
import os
import json
from datetime import datetime
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from werkzeug.utils import secure_filename
import logging
from pathlib import Path
from preprocessing.resumes.preprocess import preprocess_resume
from preprocessing.jds.scraper import LinkedInScraper
from similarity.model import SimilarityModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/api.log'),
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

RESUME_PROCESSING_TIME = Histogram(
    'resume_processing_seconds',
    'Time taken to process resumes',
    ['status']
)

JD_SCRAPING_TIME = Histogram(
    'jd_scraping_seconds',
    'Time taken to scrape job descriptions',
    ['status']
)

MATCHING_TIME = Histogram(
    'job_matching_seconds',
    'Time taken to match jobs with resumes',
    ['status']
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the project root
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
frontend_dir = os.path.join(base_dir, "frontend")
backend_dir = os.path.join(base_dir, "backend")

# Create necessary directories
os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "raw", "resumes"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "raw", "jobs"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "metadata"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "embeddings"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "resumes"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "logs"), exist_ok=True)

# Ensure directories have proper permissions
for dir_path in [
    os.path.join(data_dir, "raw"),
    os.path.join(data_dir, "raw", "resumes"),
    os.path.join(data_dir, "raw", "jobs"),
    os.path.join(data_dir, "processed"),
    os.path.join(data_dir, "metadata"),
    os.path.join(data_dir, "embeddings"),
    os.path.join(data_dir, "resumes"),
    os.path.join(data_dir, "logs")
]:
    try:
        os.chmod(dir_path, 0o777)  # Full permissions for development
    except Exception as e:
        logger.warning(f"Could not set permissions for {dir_path}: {str(e)}")

# Initialize processors
data_processor = DataProcessor(data_dir=data_dir)
model_trainer = ModelTrainer(data_dir=data_dir)
data_downloader = DataDownloader(data_dir=data_dir)

# Process existing resumes
print("Processing existing resumes...")
data_downloader.process_all_resumes()
print("Resume processing completed!")

#Process jds from linkedin
pre_process_jds = pjds.preprocess_jds()

# Add sample jobs if no jobs exist
jobs_dir = os.path.join(data_dir, "raw", "jobs")
if not os.path.exists(jobs_dir) or not os.listdir(jobs_dir):
    print("Adding sample jobs...")
    sample_jobs = [
        {
            "id": "jd_1",
            "title": "Data Scientist",
            "company": "Data Analytics Inc",
            "location": "Bangalore, India",
            "description": "We are looking for a Data Scientist to join our team. The ideal candidate will have experience with machine learning, statistical analysis, and data visualization. Skills required: Python, R, SQL, TensorFlow, PyTorch, scikit-learn, pandas, numpy, matplotlib, seaborn.",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "jd_2",
            "title": "Full Stack Developer",
            "company": "Web Solutions Ltd",
            "location": "Mumbai, India",
            "description": "We are seeking a Full Stack Developer to build web applications. The candidate should have experience with front-end and back-end technologies. Skills required: JavaScript, React, Node.js, Express, MongoDB, HTML, CSS, RESTful APIs, Git.",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "jd_3",
            "title": "Machine Learning Engineer",
            "company": "AI Innovations",
            "location": "Hyderabad, India",
            "description": "Join our AI team as a Machine Learning Engineer. You will develop and deploy machine learning models. Skills required: Python, TensorFlow, PyTorch, scikit-learn, pandas, numpy, Docker, Kubernetes, AWS, GCP, Azure.",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "jd_4",
            "title": "DevOps Engineer",
            "company": "Cloud Systems Inc",
            "location": "Pune, India",
            "description": "We are looking for a DevOps Engineer to manage our cloud infrastructure. The ideal candidate will have experience with CI/CD, containerization, and cloud platforms. Skills required: Linux, Docker, Kubernetes, Jenkins, AWS, GCP, Azure, Terraform, Ansible, Git.",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    # Save sample jobs to files
    os.makedirs(jobs_dir, exist_ok=True)
    for job in sample_jobs:
        job_file = os.path.join(jobs_dir, f"{job['id']}.json")
        with open(job_file, 'w', encoding='utf-8') as f:
            json.dump(job, f, ensure_ascii=False, indent=2)
    
    print(f"Added {len(sample_jobs)} sample jobs")

# Train the model on startup if it doesn't exist
print("Checking if model needs to be trained...")
model_path = os.path.join(base_dir, "models", "siamese_model.keras")
if not os.path.exists(model_path):
    print("Model not found. Training new model...")
    model_trainer.train_model()
    print("Model training completed!")
else:
    print("Model already exists. Skipping training.")

# Mount static files
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def read_root():
    """Serve the home page."""
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.get("/upload")
async def upload_page():
    """Serve the upload page."""
    return FileResponse(os.path.join(frontend_dir, "upload.html"))

@app.get("/matches")
async def matches_page():
    """Serve the matches page."""
    return FileResponse(os.path.join(frontend_dir, "matches.html"))

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Handle resume upload and processing."""
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='started').inc()
    
    try:
        # Validate file size (10MB limit)
        file_size = 0
        content = b""
        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        MAX_SIZE = 10 * 1024 * 1024  # 10MB
        
        while chunk := await file.read(CHUNK_SIZE):
            file_size += len(chunk)
            if file_size > MAX_SIZE:
                REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
                raise HTTPException(
                    status_code=413,
                    detail="File too large. Maximum size allowed is 10MB"
                )
            content += chunk
        
        # Reset file pointer
        await file.seek(0)
        
        # Read file content based on file type
        text_content = ""
        try:
            if file.filename.endswith('.pdf'):
                try:
                    pdf_reader = PyPDF2.PdfReader(file.file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() or ""
                except Exception as e:
                    REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing PDF file: {str(e)}"
                    )
            elif file.filename.endswith('.docx'):
                try:
                    doc = docx.Document(file.file)
                    text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                except Exception as e:
                    REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing DOCX file: {str(e)}"
                    )
            else:
                REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Please upload a PDF or DOCX file"
                )
            
            if not text_content.strip():
                REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from the uploaded file. Please ensure the file contains readable text"
                )
            
            # Save and process the resume
            resume_id = data_processor.save_resume(text_content, original_file=content)
            
            # Check if model should be retrained
            new_resumes_count = data_processor.count_new_resumes_since_last_training()
            print(f"New resumes since last training: {new_resumes_count}/5")
            
            if data_processor.should_retrain(threshold=5):
                print("Threshold of 5 new resumes reached. Scheduling model retraining...")
                # Schedule model retraining in the background
                background_tasks.add_task(retrain_model)
            else:
                print(f"Model will be retrained after {5 - new_resumes_count} more new resumes.")
            
            # Find matching jobs using the model
            match_start = time.time()
            matches = model_trainer.find_matches(resume_id)
            MATCHING_TIME.labels(status='success' if matches else 'error').observe(time.time() - match_start)
            
            # Get job details for matches
            results = []
            for job_id, score in matches:
                try:
                    job_details = data_processor.get_job_details(job_id)
                    if job_details:
                        # Ensure score is a valid float
                        score_float = float(score)
                        if not np.isnan(score_float):
                            results.append({
                                "job_id": job_id,
                                "score": score_float,
                                "title": job_details.get("title", ""),
                                "company": job_details.get("company", ""),
                                "description": job_details.get("description", ""),
                                "posting_date": job_details.get("posting_date", "")
                            })
                except (ValueError, TypeError) as e:
                    print(f"Error processing match for job {job_id}: {str(e)}")
                    continue
            
            REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='success').inc()
            REQUEST_LATENCY.labels(method='POST', endpoint='/upload-resume').observe(time.time() - start_time)
            return {"resume_id": resume_id, "matches": results}
            
        except HTTPException:
            REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
            raise
        except Exception as e:
            REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while processing the file: {str(e)}"
            )
            
    except HTTPException:
        REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/upload-resume', status='error').inc()
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/add-job")
async def add_job(job_data: dict):
    """Add a new job posting."""
    try:
        job_id = data_processor.save_job_data(job_data)
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}")
async def get_job(job_id: str):
    """Get job details by ID."""
    job_details = data_processor.get_job_details(job_id)
    if not job_details:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_details

@app.get("/resume/{resume_id}")
async def get_resume(resume_id: str):
    """Get resume details by ID."""
    resume_details = data_processor.get_resume_details(resume_id)
    if not resume_details:
        raise HTTPException(status_code=404, detail="Resume not found")
    return {"content": resume_details}

@app.get("/matches/{resume_id}")
async def get_matches(resume_id: str):
    """Get job matches for a resume."""
    start_time = time.time()
    REQUEST_COUNT.labels(method='GET', endpoint='/matches', status='started').inc()
    
    try:
        match_start = time.time()
        matches = model_trainer.find_matches(resume_id)
        MATCHING_TIME.labels(status='success' if matches else 'error').observe(time.time() - match_start)
        
        if matches:
            REQUEST_COUNT.labels(method='GET', endpoint='/matches', status='success').inc()
            REQUEST_LATENCY.labels(method='GET', endpoint='/matches').observe(time.time() - start_time)
            return {"matches": matches}
        else:
            REQUEST_COUNT.labels(method='GET', endpoint='/matches', status='error').inc()
            return {"error": "No matches found"}
            
    except Exception as e:
        logger.error(f"Error finding matches: {e}")
        REQUEST_COUNT.labels(method='GET', endpoint='/matches', status='error').inc()
        return {"error": str(e)}

def retrain_model():
    """Retrain the model in the background."""
    try:
        model_trainer.train_model()
        print("Model retraining completed successfully.")
    except Exception as e:
        print(f"Error retraining model: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.route('/resumes/upload', methods=['POST'])
def upload_resume():
    """Upload and process a resume."""
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/resumes/upload', status='started').inc()
    
    try:
        if 'resume' not in request.files:
            REQUEST_COUNT.labels(method='POST', endpoint='/resumes/upload', status='error').inc()
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            REQUEST_COUNT.labels(method='POST', endpoint='/resumes/upload', status='error').inc()
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            resume_dir = Path("data/raw/resumes")
            resume_dir.mkdir(parents=True, exist_ok=True)
            filepath = resume_dir / filename
            file.save(filepath)
            
            # Process resume
            process_start = time.time()
            success = preprocess_resume(str(filepath))
            RESUME_PROCESSING_TIME.labels(status='success' if success else 'error').observe(time.time() - process_start)
            
            if success:
                REQUEST_COUNT.labels(method='POST', endpoint='/resumes/upload', status='success').inc()
                REQUEST_LATENCY.labels(method='POST', endpoint='/resumes/upload').observe(time.time() - start_time)
                return jsonify({'message': 'Resume processed successfully', 'filename': filename}), 200
            else:
                REQUEST_COUNT.labels(method='POST', endpoint='/resumes/upload', status='error').inc()
                return jsonify({'error': 'Failed to process resume'}), 500
                
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        REQUEST_COUNT.labels(method='POST', endpoint='/resumes/upload', status='error').inc()
        return jsonify({'error': str(e)}), 500

@app.route('/scrape-linkedin-jobs', methods=['POST'])
def scrape_jobs():
    """Scrape job descriptions from LinkedIn."""
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/scrape-linkedin-jobs', status='started').inc()
    
    try:
        data = request.get_json() or {}
        keywords = data.get('keywords', '')
        location = data.get('location', 'India')
        num_pages = data.get('num_pages', 1)
        
        scraper = LinkedInScraper()
        scrape_start = time.time()
        jobs = scraper.scrape_jobs(
            search_params={"keywords": keywords, "location": location},
            num_pages=num_pages
        )
        JD_SCRAPING_TIME.labels(status='success' if jobs else 'error').observe(time.time() - scrape_start)
        
        if jobs:
            REQUEST_COUNT.labels(method='POST', endpoint='/scrape-linkedin-jobs', status='success').inc()
            REQUEST_LATENCY.labels(method='POST', endpoint='/scrape-linkedin-jobs').observe(time.time() - start_time)
            return jsonify({'message': f'Successfully scraped {len(jobs)} jobs', 'jobs': jobs}), 200
        else:
            REQUEST_COUNT.labels(method='POST', endpoint='/scrape-linkedin-jobs', status='error').inc()
            return jsonify({'error': 'Failed to scrape jobs'}), 500
            
    except Exception as e:
        logger.error(f"Error scraping jobs: {e}")
        REQUEST_COUNT.labels(method='POST', endpoint='/scrape-linkedin-jobs', status='error').inc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
