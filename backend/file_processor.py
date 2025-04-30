import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime

class FileResumeProcessor:
    def __init__(self, data_dir="data"):
        """Initialize the processor with file-based storage."""
        self.data_dir = data_dir
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize or load vectorizer
        self._initialize_vectorizer()
    
    def _create_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            os.path.join(self.data_dir, "raw"),
            os.path.join(self.data_dir, "raw", "resumes"),
            os.path.join(self.data_dir, "raw", "jobs"),
            os.path.join(self.data_dir, "processed"),
            os.path.join(self.data_dir, "embeddings")
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_vectorizer(self):
        """Initialize or load the TF-IDF vectorizer."""
        vectorizer_path = os.path.join(self.data_dir, "processed", "vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
    
    def preprocess_text(self, text):
        """Basic text preprocessing."""
        if not isinstance(text, str):
            return ""
        return text.lower().strip()
    
    def save_job_data(self, job_data, job_id=None):
        """Save job data to file system."""
        if job_id is None:
            job_id = str(datetime.now().timestamp())
        
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
        
        # Save vectorizer if it's new
        if not os.path.exists(os.path.join(self.data_dir, "processed", "vectorizer.pkl")):
            with open(os.path.join(self.data_dir, "processed", "vectorizer.pkl"), 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        return job_id
    
    def save_resume(self, resume_text, resume_id=None):
        """Save resume data to file system."""
        if resume_id is None:
            resume_id = str(datetime.now().timestamp())
        
        # Save raw resume
        resume_path = os.path.join(self.data_dir, "raw", "resumes", f"{resume_id}.txt")
        with open(resume_path, 'w', encoding='utf-8') as f:
            f.write(resume_text)
        
        # Process and save resume embedding
        processed_text = self.preprocess_text(resume_text)
        resume_embedding = self.vectorizer.transform([processed_text])
        
        # Save embedding
        embedding_path = os.path.join(self.data_dir, "embeddings", f"resume_{resume_id}.npz")
        np.savez(embedding_path, data=resume_embedding.toarray())
        
        return resume_id
    
    def find_matches(self, resume_id, top_k=5):
        """Find matching jobs for a given resume."""
        # Load resume embedding
        resume_embedding_path = os.path.join(self.data_dir, "embeddings", f"resume_{resume_id}.npz")
        resume_embedding = np.load(resume_embedding_path)['data']
        
        # Get all job embeddings
        job_embeddings = []
        job_ids = []
        
        for filename in os.listdir(os.path.join(self.data_dir, "embeddings")):
            if filename.startswith("job_") and filename.endswith(".npz"):
                job_id = filename[4:-4]  # Remove 'job_' prefix and '.npz' suffix
                job_embedding_path = os.path.join(self.data_dir, "embeddings", filename)
                job_embedding = np.load(job_embedding_path)['data']
                
                job_embeddings.append(job_embedding)
                job_ids.append(job_id)
        
        if not job_embeddings:
            return []
        
        # Calculate similarities
        job_embeddings = np.vstack(job_embeddings)
        similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        matches = [(job_ids[i], float(similarities[i])) for i in top_indices]
        
        return matches
    
    def get_job_details(self, job_id):
        """Retrieve job details from storage."""
        job_path = os.path.join(self.data_dir, "raw", "jobs", f"{job_id}.json")
        if os.path.exists(job_path):
            with open(job_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None 