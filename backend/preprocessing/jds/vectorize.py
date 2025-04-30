import os
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
#from ..database import DatabaseManager
import json

UTC=__import__("datetime").timezone.utc
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JDVectorizer:
    """Class to handle vectorization of job descriptions using TF-IDF."""
    
    def __init__(self, base_dir="data", max_features=10000, ngram_range=(1, 2)):
        """
        Initialize the TF-IDF vectorizer for job descriptions.
        
        Args:
            base_dir (str): Base directory for storing files
            max_features (int): Maximum number of features (terms) to keep
            ngram_range (tuple): Range of n-gram sizes to extract
        """
        self.base_dir = Path(base_dir)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.is_fitted = False
        #self.db = DatabaseManager()
        
        # Create necessary directories
        self.vectors_dir = self.base_dir / "embeddings"
        self.metadata_dir = self.base_dir / "metadata"
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(self, texts):
        """
        Fit the vectorizer on a list of texts.
        
        Args:
            texts (list): List of preprocessed text documents
        """
        try:
            self.vectorizer.fit(texts)
            self.is_fitted = True
            logger.info(f"TF-IDF vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
        except Exception as e:
            logger.error(f"Error fitting TF-IDF vectorizer: {e}")
            raise
    
    def transform(self, text):
        """
        Transform a single text document into a TF-IDF vector.
        
        Args:
            text (str): Preprocessed text document
            
        Returns:
            numpy.ndarray: TF-IDF vector
        """
        try:
            if not self.is_fitted:
                raise ValueError("Vectorizer must be fitted before transforming")
            
            # Transform the text
            vector = self.vectorizer.transform([text]).toarray()[0]
            
            return vector
        except Exception as e:
            logger.error(f"Error transforming text: {e}")
            return None
    
    def save(self, file_path):
        """
        Save the fitted vectorizer to a file.
        
        Args:
            file_path (str): Path to save the vectorizer
        """
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self.vectorizer, file)
            logger.info(f"TF-IDF vectorizer saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving TF-IDF vectorizer: {e}")
    
    def load(self, file_path):
        """
        Load a fitted vectorizer from a file.
        
        Args:
            file_path (str): Path to the saved vectorizer
        """
        try:
            with open(file_path, 'rb') as file:
                self.vectorizer = pickle.load(file)
            self.is_fitted = True
            logger.info(f"TF-IDF vectorizer loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading TF-IDF vectorizer: {e}")
            raise
    
    def vectorize_job_description(self, job_id: str, job_data: dict) -> bool:
        """
        Vectorize a job description and save to files.
        
        Args:
            job_id (str): Unique identifier for the job description
            job_data (dict): Job description data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get preprocessed text
            text = job_data.get("description", "")
            if not text:
                logger.error(f"No description found for job: {job_id}")
                return False
            
            # Get document embedding
            vector = self.transform(text)
            
            if vector is not None:
                # Save vector
                vector_path = self.vectors_dir / f"job_{job_id}.npz"
                np.savez(vector_path, data=vector)
                
                # Save metadata
                metadata = {
                    "id": job_id,
                    "title": job_data.get("title", ""),
                    "company": job_data.get("company", ""),
                    "posting_date": job_data.get("posting_date", datetime.now(UTC).isoformat()),
                    "embedding_path": f"job_{job_id}.npz"
                }
                metadata_path = self.metadata_dir / f"job_{job_id}.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Vectorized job description saved: {job_id}")
                return True
            else:
                logger.error(f"Failed to vectorize job description: {job_id}")
                return False
        except Exception as e:
            logger.error(f"Error vectorizing job description {job_id}: {e}")
            return False
    
    def vectorize_jobs_from_directory(self, jobs_dir: str) -> dict:
        """
        Vectorize all job descriptions from a directory.
        
        Args:
            jobs_dir (str): Directory containing job description JSON files
            
        Returns:
            dict: Dictionary mapping job description IDs to success status
        """
        results = {}
        jobs_dir = Path(jobs_dir)
        
        if not jobs_dir.exists():
            logger.error(f"Jobs directory not found: {jobs_dir}")
            return results
        
        # Collect all texts to fit the vectorizer
        texts = []
        valid_jobs = []
        
        for job_file in jobs_dir.glob("*.json"):
            try:
                with open(job_file, 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
                    text = job_data.get("description", "")
                    if text:
                        texts.append(text)
                        valid_jobs.append((job_file.stem, job_data))
            except Exception as e:
                logger.error(f"Error reading job file {job_file}: {e}")
                continue
        
        if not texts:
            logger.warning("No valid job descriptions found for vectorization")
            return results
        
        # Fit the vectorizer on all texts
        self.fit(texts)
        
        # Vectorize each job description
        for job_id, job_data in valid_jobs:
            success = self.vectorize_job_description(job_id, job_data)
            results[job_id] = success
        
        return results

class FileHandler:
    def __init__(self, base_dir):
        """
        Initialize file handler.
        
        Args:
            base_dir (Path): Base directory for storing files
        """
        self.base_dir = Path(base_dir)
        self.vectors_dir = self.base_dir / "vectors"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create necessary directories
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def save_vector(self, jd_id, vector, metadata=None):
        """
        Save job description vector to file.
        
        Args:
            jd_id (str): Unique identifier for the job description
            vector (numpy.ndarray): Job description vector
            metadata (dict): Additional metadata about the job description
        """
        try:
            # Save vector
            vector_path = self.vectors_dir / f"{jd_id}_vector.npy"
            np.save(vector_path, vector)
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{jd_id}_metadata.json"
            metadata = metadata or {}
            metadata.update({
                "jd_id": jd_id,
                "dimension": len(vector),
                "timestamp": datetime.now(UTC).isoformat()
            })
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved vector and metadata for job description {jd_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector for job description {jd_id}: {e}")
            return False
    
    def get_vector(self, jd_id):
        """
        Get job description vector from file.
        
        Args:
            jd_id (str): Unique identifier for the job description
            
        Returns:
            numpy.ndarray: Job description vector
        """
        try:
            vector_path = self.vectors_dir / f"{jd_id}_vector.npy"
            if vector_path.exists():
                return np.load(vector_path)
            else:
                logger.warning(f"Vector not found for job description {jd_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting vector for job description {jd_id}: {e}")
            return None
    
    def get_metadata(self, jd_id):
        """
        Get job description metadata from file.
        
        Args:
            jd_id (str): Unique identifier for the job description
            
        Returns:
            dict: Job description metadata
        """
        try:
            metadata_path = self.metadata_dir / f"{jd_id}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Metadata not found for job description {jd_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting metadata for job description {jd_id}: {e}")
            return None

def vectorize_jd(jd_id, text, vectorizer, file_handler):
    """
    Vectorize a job description and save to files.
    
    Args:
        jd_id (str): Unique identifier for the job description
        text (str): Preprocessed job description text
        vectorizer (JDVectorizer): TF-IDF vectorizer instance
        file_handler (FileHandler): File handler instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get document embedding
        vector = vectorizer.transform(text)
        
        if vector is not None:
            # Prepare metadata
            metadata = {
                "jd_id": jd_id,
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            # Save to files
            success = file_handler.save_vector(jd_id, vector, metadata)
            
            return success
        else:
            logger.error(f"Failed to vectorize job description: {jd_id}")
            return False
    except Exception as e:
        logger.error(f"Error vectorizing job description {jd_id}: {e}")
        return False

def vectorize_jds(jds, file_handler=None):
    """
    Vectorize all job descriptions.
    
    Args:
        jds (dict): Dictionary mapping job description IDs to preprocessed texts
        file_handler (FileHandler, optional): File handler instance
        
    Returns:
        dict: Dictionary mapping job description IDs to (vector, metadata) tuples
    """
    try:
        # Initialize vectorizer
        vectorizer = JDVectorizer(max_features=10000, ngram_range=(1, 2))
        
        # Collect all texts to fit the vectorizer
        texts = list(jds.values())
        jd_ids = list(jds.keys())
        
        # Fit the vectorizer on all texts
        vectorizer.fit(texts)
        
        # Vectorize each job description
        results = {}
        for jd_id, text in jds.items():
            vector = vectorizer.transform(text)
            if vector is not None:
                metadata = {
                    "jd_id": jd_id,
                    "timestamp": datetime.now(UTC).isoformat()
                }
                results[jd_id] = (vector, metadata)
                
                # Save to files if file_handler is provided
                if file_handler:
                    file_handler.save_vector(jd_id, vector, metadata)
        
        logger.info(f"Vectorized {len(results)} job descriptions")
        return results
    except Exception as e:
        logger.error(f"Error vectorizing job descriptions: {e}")
        return {}

def main():
    """Main function to test job description vectorization."""
    try:
        # Initialize vectorizer with custom base directory
        vectorizer = JDVectorizer(base_dir="custom_data")
        
        # Vectorize jobs from a specific directory
        results = vectorizer.vectorize_jobs_from_directory("path/to/jobs")
        
        # Log results
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Vectorized {len(results)} job descriptions")
        logger.info(f"Success: {success_count}")
    except Exception as e:
        logger.error(f"Error in job description vectorization process: {e}")

if __name__ == "__main__":
    main() 
