import re
import nltk
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Open Multilingual Wordnet

class JDPreprocessor:
    """Class to handle preprocessing of job descriptions."""
    
    def __init__(self, data_dir=Path(__file__).parent.parent.parent / "data" / "raw_jds" / "json", custom_stopwords=None):
        """
        Initialize the JDPreprocessor.
        
        Args:
            data_dir (str): Path to the data directory
            custom_stopwords (list, optional): List of additional stopwords to remove
        """
        print("jdprocessor init: " , data_dir)
    	#self.data_dir = data_dir
        self.raw_jobs_dir =  data_dir #os.path.join(data_dir, "raw", "jobs")
        self.processed_jobs_dir = os.path.join(data_dir, "processed", "jobs")
        os.makedirs(self.processed_jobs_dir, exist_ok=True)
        
        self.custom_stopwords = custom_stopwords or [
            'job', 'position', 'role', 'opportunity', 'career', 'company', 'organization',
            'looking', 'seeking', 'hiring', 'join', 'join us', 'join our', 'join the',
            'apply', 'application', 'submit', 'submission', 'send', 'email', 'contact',
            'phone', 'address', 'location', 'remote', 'hybrid', 'onsite', 'in-office',
            'full-time', 'part-time', 'contract', 'temporary', 'permanent', 'salary',
            'compensation', 'benefits', 'package', 'offer', 'competitive', 'market',
            'rate', 'hourly', 'annual', 'yearly', 'monthly', 'weekly', 'daily'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning, removing stopwords, and lemmatizing.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Remove stopwords
            text_without_stopwords = self._remove_stopwords(cleaned_text)
            
            # Lemmatize
            lemmatized_text = self._lemmatize_text(text_without_stopwords)
            
            return lemmatized_text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters, extra whitespace, etc."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        # Get default stopwords
        stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords
        stop_words.update(self.custom_stopwords)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(filtered_tokens)
    
    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize text using WordNet lemmatizer."""
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(lemmatized_tokens)
    
    def preprocess_job_description(self, job_id: str) -> bool:
        """
        Preprocess a job description and save to file.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get job description from file
            job_file = os.path.join(self.raw_jobs_dir, f"{job_id}.json")
            if not os.path.exists(job_file):
                logger.error(f"Job description not found: {job_id}")
                return False
            
            with open(job_file, 'r', encoding='utf-8') as f:
                job = json.load(f)
            
            # Combine description and requirements
            full_text = job.get("description", "")
            requirements = job.get("requirements", [])
            if requirements:
                full_text += " " + " ".join(requirements)
            
            # Preprocess the text
            preprocessed_text = self.preprocess_text(full_text)
            
            if preprocessed_text:
                # Update job description with preprocessed text
                job["preprocessed_text"] = preprocessed_text
                
                # Save to processed directory
                processed_file = os.path.join(self.processed_jobs_dir, f"{job_id}.json")
                with open(processed_file, 'w', encoding='utf-8') as f:
                    json.dump(job, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Preprocessed job description saved: {job_id}")
                return True
            else:
                logger.error(f"Failed to preprocess job description: {job_id}")
                return False
        except Exception as e:
            logger.error(f"Error preprocessing job description {job_id}: {e}")
            return False
    
    def preprocess_all_job_descriptions(self, limit: int = None) -> dict:
        """
        Preprocess all job descriptions in the raw directory.
        
        Args:
            limit (int, optional): Maximum number of job descriptions to process
            
        Returns:
            dict: Dictionary mapping job IDs to success status
        """
        results = {}
        
        # Get all job description files
        job_files = [f for f in os.listdir(self.raw_jobs_dir) if f.endswith('.json')]
        if limit:
            job_files = job_files[:limit]
        
        # Process each job description
        for job_file in job_files:
            job_id = os.path.splitext(job_file)[0]
            success = self.preprocess_job_description(job_id)
            results[job_id] = success
        
        return results

def main():
    """Main function to test job description preprocessing."""
    try:
        # Initialize preprocessor
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        preprocessor = JDPreprocessor(data_dir=data_dir)
        
        # Preprocess all job descriptions
        results = preprocessor.preprocess_all_job_descriptions(limit=10)
        
        # Log results
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Preprocessed {len(results)} job descriptions")
        logger.info(f"Success: {success_count}")
    except Exception as e:
        logger.error(f"Error in job description preprocessing process: {e}")

if __name__ == "__main__":
    main() 
