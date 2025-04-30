import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from datetime import datetime
import random

class DataProcessor:
    def __init__(self, data_dir="data"):
        """Initialize the data processor."""
        # Use the absolute path to the data directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.data_dir = os.path.join(base_dir, data_dir)
        self.raw_resumes_dir = os.path.join(self.data_dir, "raw", "resumes")
        self.pdf_resumes_dir = os.path.join(self.data_dir, "resumes", "data", "data")
        self.raw_jobs_dir = os.path.join(self.data_dir, "raw", "jobs")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.metadata_dir = os.path.join(self.data_dir, "metadata")
        self.embeddings_dir = os.path.join(self.data_dir, "embeddings")
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.raw_resumes_dir, exist_ok=True)
        os.makedirs(self.pdf_resumes_dir, exist_ok=True)
        os.makedirs(self.raw_jobs_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize or load vectorizer
        self._initialize_vectorizer()
        
        # Add sample jobs if no jobs exist
        if not self._has_jobs():
            self._add_sample_jobs()
    
    def _create_directories(self):
        """Create necessary directories for data storage."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_dir, "data")
        
        directories = [
            os.path.join(data_dir, "raw"),
            os.path.join(data_dir, "raw", "resumes"),
            os.path.join(data_dir, "raw", "jobs"),
            os.path.join(data_dir, "processed"),
            os.path.join(data_dir, "embeddings"),
            os.path.join(data_dir, "metadata")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Set the data directory path
        self.data_dir = data_dir
    
    def _has_jobs(self):
        """Check if there are any jobs in the system."""
        job_files = [f for f in os.listdir(os.path.join(self.data_dir, "raw", "jobs")) 
                    if f.endswith('.json')]
        return len(job_files) > 0
    
    def _add_sample_jobs(self):
        """Add sample job data to initialize the system."""
        sample_jobs = [
            {
                "title": "Software Engineer",
                "company": "Tech Corp",
                "description": "We are looking for a Software Engineer with experience in Python, JavaScript, and web development. The ideal candidate should have strong problem-solving skills and experience with modern development practices. Required skills: Python, JavaScript, React, Node.js, SQL. Nice to have: AWS, Docker, Kubernetes.",
                "posting_date": datetime.now().isoformat()
            },
            {
                "title": "Data Scientist",
                "company": "Data Analytics Inc",
                "description": "Seeking a Data Scientist with expertise in machine learning, Python, and data analysis. Experience with deep learning frameworks and big data technologies is a plus. Required skills: Python, Machine Learning, SQL, Statistics. Nice to have: TensorFlow, PyTorch, Spark.",
                "posting_date": datetime.now().isoformat()
            },
            {
                "title": "Full Stack Developer",
                "company": "Web Solutions Ltd",
                "description": "Looking for a Full Stack Developer with experience in React, Node.js, and database design. Knowledge of cloud platforms and DevOps practices is required. Required skills: JavaScript, React, Node.js, MongoDB, AWS. Nice to have: Docker, CI/CD, GraphQL.",
                "posting_date": datetime.now().isoformat()
            },
            {
                "title": "Machine Learning Engineer",
                "company": "AI Innovations",
                "description": "Join our AI team to build and deploy machine learning models. Experience with deep learning, NLP, and MLOps required. Required skills: Python, TensorFlow/PyTorch, ML, Git. Nice to have: Kubernetes, AWS, Docker.",
                "posting_date": datetime.now().isoformat()
            },
            {
                "title": "DevOps Engineer",
                "company": "Cloud Systems Inc",
                "description": "Looking for a DevOps Engineer to manage our cloud infrastructure and CI/CD pipelines. Required skills: AWS, Docker, Kubernetes, Jenkins, Linux. Nice to have: Terraform, Ansible, Python.",
                "posting_date": datetime.now().isoformat()
            }
        ]
        
        print("Adding sample jobs...")
        for job in sample_jobs:
            job_id = self.save_job_data(job)
            print(f"Added job: {job['title']} at {job['company']} (ID: {job_id})")
    
    def _initialize_vectorizer(self):
        """Initialize or load the TF-IDF vectorizer."""
        vectorizer_path = os.path.join(self.data_dir, "processed", "vectorizer.pkl")
        
        # Ensure the processed directory exists
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            # Initialize with empty text to create vocabulary
            self.vectorizer.fit(["initialization text"])
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
    
    def preprocess_text(self, text):
        """Basic text preprocessing."""
        if not isinstance(text, str):
            return ""
        return text.lower().strip()
    
    def save_job_data(self, job_data, job_id=None):
        """Save job data to file system."""
        if job_id is None:
            job_id = str(datetime.now().timestamp())
        
        # Add posting date if not present
        if 'posting_date' not in job_data:
            job_data['posting_date'] = datetime.now().isoformat()
        
        # Save raw job data
        job_path = os.path.join(self.data_dir, "raw", "jobs", f"{job_id}.json")
        with open(job_path, 'w', encoding='utf-8') as f:
            json.dump(job_data, f, ensure_ascii=False, indent=2)
        
        # Process and save job embedding
        processed_text = self.preprocess_text(job_data.get('description', ''))
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit([processed_text])
        
        job_embedding = self.vectorizer.transform([processed_text])
        
        # Save embedding
        embedding_path = os.path.join(self.data_dir, "embeddings", f"job_{job_id}.npz")
        np.savez(embedding_path, data=job_embedding.toarray())
        
        # Save metadata
        metadata = {
            'id': job_id,
            'title': job_data.get('title', ''),
            'company': job_data.get('company', ''),
            'posting_date': job_data.get('posting_date', ''),
            'embedding_path': f"job_{job_id}.npz"
        }
        metadata_path = os.path.join(self.data_dir, "metadata", f"job_{job_id}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Save vectorizer if it's new
        if not os.path.exists(os.path.join(self.data_dir, "processed", "vectorizer.pkl")):
            with open(os.path.join(self.data_dir, "processed", "vectorizer.pkl"), 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        return job_id
    
    def save_resume(self, resume_text, resume_id=None, original_file=None):
        """Save resume data to file system."""
        if resume_id is None:
            resume_id = str(datetime.now().timestamp())
        
        # Save raw resume text
        resume_text_path = os.path.join(self.raw_resumes_dir, f"{resume_id}.txt")
        with open(resume_text_path, 'w', encoding='utf-8') as f:
            f.write(resume_text)
        
        # Save original PDF if provided
        if original_file:
            resume_pdf_path = os.path.join(self.pdf_resumes_dir, f"{resume_id}.pdf")
            with open(resume_pdf_path, 'wb') as f:
                f.write(original_file)
        
        # Process and save resume embedding
        processed_text = self.preprocess_text(resume_text)
        resume_embedding = self.vectorizer.transform([processed_text])
        
        # Save embedding
        embedding_path = os.path.join(self.embeddings_dir, f"resume_{resume_id}.npz")
        np.savez(embedding_path, data=resume_embedding.toarray())
        
        # Save metadata
        metadata = {
            'id': resume_id,
            'upload_date': datetime.now().isoformat(),
            'embedding_path': f"resume_{resume_id}.npz",
            'has_pdf': original_file is not None,
            'pdf_path': f"{resume_id}.pdf" if original_file else None,
            'text_path': f"{resume_id}.txt"
        }
        metadata_path = os.path.join(self.metadata_dir, f"resume_{resume_id}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return resume_id
    
    def get_resume_embedding(self, resume_id):
        """Get the embedding for a resume."""
        embedding_path = os.path.join(self.data_dir, "embeddings", f"resume_{resume_id}.npz")
        if os.path.exists(embedding_path):
            return np.load(embedding_path)['data']
        return None
    
    def get_job_embedding(self, job_id):
        """Get the embedding for a job."""
        embedding_path = os.path.join(self.data_dir, "embeddings", f"job_{job_id}.npz")
        if os.path.exists(embedding_path):
            return np.load(embedding_path)['data']
        return None
    
    def get_job_details(self, job_id):
        """Retrieve job details from storage."""
        job_path = os.path.join(self.data_dir, "raw", "jobs", f"{job_id}.json")
        if os.path.exists(job_path):
            with open(job_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_resume_details(self, resume_id):
        """Retrieve resume details from storage."""
        resume_path = os.path.join(self.data_dir, "raw", "resumes", f"{resume_id}.txt")
        if os.path.exists(resume_path):
            with open(resume_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def get_recent_jobs(self, limit=50):
        """Get the most recent jobs."""
        job_metadata_files = [f for f in os.listdir(os.path.join(self.data_dir, "metadata")) 
                             if f.startswith("job_") and f.endswith(".json")]
        
        job_metadata = []
        for file in job_metadata_files:
            with open(os.path.join(self.data_dir, "metadata", file), 'r', encoding='utf-8') as f:
                job_metadata.append(json.load(f))
        
        # Sort by posting date (most recent first)
        job_metadata.sort(key=lambda x: x.get('posting_date', ''), reverse=True)
        
        return job_metadata[:limit]
    
    def prepare_training_data(self, num_samples=1000):
        """Prepare training data for the Siamese network."""
        print("\nPreparing training data...")
        
        # Get all resume and job embeddings
        resume_embeddings = []
        resume_ids = []
        job_embeddings = []
        job_ids = []
        
        # Load resume embeddings
        print("Loading resume embeddings...")
        resume_files = [f for f in os.listdir(os.path.join(self.data_dir, "embeddings")) 
                       if f.startswith("resume_") and f.endswith(".npz")]
        print(f"Found {len(resume_files)} resume embeddings")
        
        for filename in resume_files:
            resume_id = filename[7:-4]  # Remove 'resume_' prefix and '.npz' suffix
            embedding_path = os.path.join(self.data_dir, "embeddings", filename)
            embedding = np.load(embedding_path)['data']
            
            resume_embeddings.append(embedding)
            resume_ids.append(resume_id)
            print(f"Loaded resume embedding for ID: {resume_id}")
        
        # Load job embeddings
        print("\nLoading job embeddings...")
        job_files = [f for f in os.listdir(os.path.join(self.data_dir, "embeddings")) 
                    if f.startswith("job_") and f.endswith(".npz")]
        print(f"Found {len(job_files)} job embeddings")
        
        for filename in job_files:
            job_id = filename[4:-4]  # Remove 'job_' prefix and '.npz' suffix
            embedding_path = os.path.join(self.data_dir, "embeddings", filename)
            embedding = np.load(embedding_path)['data']
            
            job_embeddings.append(embedding)
            job_ids.append(job_id)
            print(f"Loaded job embedding for ID: {job_id}")
        
        if not resume_embeddings or not job_embeddings:
            print("No embeddings found for training!")
            return None, None, None, None
        
        print(f"\nTotal resumes: {len(resume_embeddings)}")
        print(f"Total jobs: {len(job_embeddings)}")
        
        # Convert to numpy arrays
        resume_embeddings = np.vstack(resume_embeddings)
        job_embeddings = np.vstack(job_embeddings)
        print(f"Resume embeddings shape: {resume_embeddings.shape}")
        print(f"Job embeddings shape: {job_embeddings.shape}")
        
        # Generate positive and negative pairs
        print("\nGenerating training pairs...")
        X = []
        y = []
        
        # Positive pairs (resume-job matches)
        num_positive = min(len(resume_embeddings), len(job_embeddings), num_samples // 2)
        print(f"Generating {num_positive} positive pairs")
        
        for i in range(num_positive):
            X.append([resume_embeddings[i], job_embeddings[i]])
            y.append(1)  # Positive match
        
        # Negative pairs (non-matches)
        num_negative = num_samples // 2
        print(f"Generating {num_negative} negative pairs")
        
        for _ in range(num_negative):
            resume_idx = random.randint(0, len(resume_embeddings) - 1)
            job_idx = random.randint(0, len(job_embeddings) - 1)
            
            # Ensure we don't create a positive pair
            while resume_idx == job_idx and len(resume_embeddings) == len(job_embeddings):
                job_idx = random.randint(0, len(job_embeddings) - 1)
            
            X.append([resume_embeddings[resume_idx], job_embeddings[job_idx]])
            y.append(0)  # Negative match
        
        # Shuffle the data
        print("\nShuffling data...")
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = np.array(X)[indices]
        y = np.array(y)[indices]
        
        # Split into train and validation sets
        split_idx = int(0.8 * len(X))
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Validation labels shape: {y_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def should_retrain(self, threshold=5):
        """Check if model should be retrained based on new resumes."""
        resume_metadata_files = [f for f in os.listdir(self.metadata_dir) 
                               if f.startswith("resume_") and f.endswith(".json")]
        
        # Count resumes added since last training
        last_training_path = os.path.join(self.processed_dir, "last_training.txt")
        if not os.path.exists(last_training_path):
            return len(resume_metadata_files) >= threshold
        
        with open(last_training_path, 'r') as f:
            last_training_time = f.read().strip()
        
        new_resumes = 0
        for file in resume_metadata_files:
            with open(os.path.join(self.metadata_dir, file), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if metadata.get('upload_date', '') > last_training_time:
                    new_resumes += 1
        
        return new_resumes >= threshold
    
    def update_last_training_time(self):
        """Update the last training time."""
        last_training_path = os.path.join(self.processed_dir, "last_training.txt")
        with open(last_training_path, 'w') as f:
            f.write(datetime.now().isoformat())
    
    def count_new_resumes_since_last_training(self):
        """Count the number of new resumes added since the last training."""
        resume_metadata_files = [f for f in os.listdir(self.metadata_dir) 
                               if f.startswith("resume_") and f.endswith(".json")]
        
        # If no last training time, count all resumes
        last_training_path = os.path.join(self.processed_dir, "last_training.txt")
        if not os.path.exists(last_training_path):
            return len(resume_metadata_files)
        
        with open(last_training_path, 'r') as f:
            last_training_time = f.read().strip()
        
        new_resumes = 0
        for file in resume_metadata_files:
            with open(os.path.join(self.metadata_dir, file), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if metadata.get('upload_date', '') > last_training_time:
                    new_resumes += 1
        
        return new_resumes 