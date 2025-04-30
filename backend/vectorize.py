import os
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

UTC=__import__("datetime").timezone.utc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResumeVectorizer:
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        """
        Initialize the TF-IDF vectorizer.
        
        Args:
            max_features (int): Maximum number of features (terms) to keep
            ngram_range (tuple): Range of n-gram sizes to extract
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.is_fitted = False
    
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
    
    def save_vector(self, resume_id, vector, metadata=None):
        """
        Save resume vector to file.
        
        Args:
            resume_id (str): Unique identifier for the resume
            vector (numpy.ndarray): Resume vector
            metadata (dict): Additional metadata about the resume
        """
        try:
            # Save vector
            vector_path = self.vectors_dir / f"{resume_id}_vector.npy"
            np.save(vector_path, vector)
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{resume_id}_metadata.json"
            metadata = metadata or {}
            metadata.update({
                "resume_id": resume_id,
                "dimension": len(vector),
                "timestamp": datetime.now(UTC).isoformat()
            })
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved vector and metadata for resume {resume_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector for resume {resume_id}: {e}")
            return False
    
    def get_vector(self, resume_id):
        """
        Get resume vector from file.
        
        Args:
            resume_id (str): Unique identifier for the resume
            
        Returns:
            numpy.ndarray: Resume vector
        """
        try:
            vector_path = self.vectors_dir / f"{resume_id}_vector.npy"
            if vector_path.exists():
                return np.load(vector_path)
            else:
                logger.warning(f"Vector not found for resume {resume_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting vector for resume {resume_id}: {e}")
            return None
    
    def get_metadata(self, resume_id):
        """
        Get resume metadata from file.
        
        Args:
            resume_id (str): Unique identifier for the resume
            
        Returns:
            dict: Resume metadata
        """
        try:
            metadata_path = self.metadata_dir / f"{resume_id}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Metadata not found for resume {resume_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting metadata for resume {resume_id}: {e}")
            return None

def vectorize_resume_file(file_path, vectorizer, file_handler):
    """
    Vectorize a resume file and save to files.
    
    Args:
        file_path (str): Path to preprocessed resume file
        vectorizer (ResumeVectorizer): TF-IDF vectorizer instance
        file_handler (FileHandler): File handler instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the preprocessed text
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Get document embedding
        vector = vectorizer.transform(text)
        
        if vector is not None:
            # Generate resume ID from file path
            resume_id = str(Path(file_path).stem)
            
            # Prepare metadata
            metadata = {
                "file_path": str(file_path),
                "file_name": Path(file_path).name,
                "file_type": Path(file_path).suffix
            }
            
            # Save to files
            success = file_handler.save_vector(resume_id, vector, metadata)
            
            return success
        else:
            logger.error(f"Failed to vectorize resume: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error vectorizing resume file {file_path}: {e}")
        return False

def vectorize_resume_directory(directory_path, vectorizer, file_handler):
    """
    Vectorize all resume files in a directory.
    
    Args:
        directory_path (str): Path to directory with preprocessed resumes
        vectorizer (ResumeVectorizer): TF-IDF vectorizer instance
        file_handler (FileHandler): File handler instance
        
    Returns:
        dict: Dictionary mapping file paths to success status
    """
    directory_path = Path(directory_path)
    results = {}
    
    # Look for preprocessed resumes
    preprocessed_dir = directory_path / "preprocessed"
    if preprocessed_dir.exists():
        # First, collect all texts to fit the vectorizer
        texts = []
        file_paths = []
        
        for file_path in preprocessed_dir.glob("*_preprocessed.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                texts.append(text)
                file_paths.append(file_path)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        # Fit the vectorizer on all texts
        if texts:
            vectorizer.fit(texts)
            
            # Now vectorize each file
            for file_path in file_paths:
                success = vectorize_resume_file(file_path, vectorizer, file_handler)
                results[str(file_path)] = success
        else:
            logger.warning(f"No preprocessed resumes found in {preprocessed_dir}")
    else:
        logger.warning(f"Preprocessed directory not found: {preprocessed_dir}")
    
    return results

def main():
    """Main function to test vectorization."""
    # Example usage
    base_dir = Path(__file__).parent.parent.parent.parent
    raw_resumes_dir = base_dir / "data" / "raw" / "resumes"
    vectors_dir = base_dir / "data" / "vectors"
    
    try:
        # Initialize TF-IDF vectorizer
        vectorizer = ResumeVectorizer(max_features=10000, ngram_range=(1, 2))
        
        # Initialize file handler
        file_handler = FileHandler(vectors_dir)
        
        if raw_resumes_dir.exists():
            results = vectorize_resume_directory(raw_resumes_dir, vectorizer, file_handler)
            logger.info(f"Vectorized {len(results)} resume files")
            logger.info(f"Success: {sum(1 for success in results.values() if success)}")
            
            # Save the fitted vectorizer
            vectorizer_path = vectors_dir / "tfidf_vectorizer.pkl"
            vectorizer.save(vectorizer_path)
        else:
            logger.error(f"Directory not found: {raw_resumes_dir}")
    except Exception as e:
        logger.error(f"Error in vectorization process: {e}")

if __name__ == "__main__":
    main() 