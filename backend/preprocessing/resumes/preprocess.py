import re
import nltk
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

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

class ResumePreprocessor:
    """Class to handle preprocessing of resume text."""
    
    def __init__(self, custom_stopwords=None):
        """
        Initialize the ResumePreprocessor.
        
        Args:
            custom_stopwords (list, optional): List of additional stopwords to remove
        """
        self.custom_stopwords = custom_stopwords or [
            'resume', 'cv', 'curriculum', 'vitae', 'page', 'of', 'the', 'and', 'a', 'an',
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'up', 'about', 'into',
            'over', 'after', 'email', 'phone', 'address', 'contact', 'information'
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

def clean_text(text):
    """Clean text by removing special characters, extra whitespace, etc."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text, custom_stopwords=None):
    """Remove stopwords from text."""
    # Get default stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords if provided
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    """Lemmatize text using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(lemmatized_tokens)

def preprocess_text(text, custom_stopwords=None):
    """Preprocess text by cleaning, removing stopwords, and lemmatizing."""
    try:
        # Clean text
        cleaned_text = clean_text(text)
        
        # Remove stopwords
        text_without_stopwords = remove_stopwords(cleaned_text, custom_stopwords)
        
        # Lemmatize
        lemmatized_text = lemmatize_text(text_without_stopwords)
        
        return lemmatized_text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return None

def preprocess_resume_file(file_path, custom_stopwords=None):
    """Preprocess a resume file."""
    try:
        # Read the extracted text file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        
        # Preprocess the text
        preprocessed_text = preprocess_text(text, custom_stopwords)
        
        if preprocessed_text:
            # Save preprocessed text to a new file
            output_path = Path(file_path).parent.parent / "preprocessed" / f"{Path(file_path).stem}_preprocessed.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(preprocessed_text)
            
            logger.info(f"Preprocessed resume saved to {output_path}")
            return str(output_path)
        else:
            logger.error(f"Failed to preprocess resume: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error preprocessing resume file {file_path}: {e}")
        return None

def preprocess_resume_directory(directory_path, custom_stopwords=None):
    """Preprocess all resume files in a directory."""
    directory_path = Path(directory_path)
    results = {}
    
    # Look for extracted text files
    extracted_dir = directory_path / "extracted"
    if extracted_dir.exists():
        for file_path in extracted_dir.glob("*_extracted.txt"):
            preprocessed_path = preprocess_resume_file(file_path, custom_stopwords)
            if preprocessed_path:
                results[str(file_path)] = preprocessed_path
    else:
        logger.warning(f"Extracted directory not found: {extracted_dir}")
    
    return results

def main():
    """Main function to test preprocessing."""
    # Example usage
    base_dir = Path(__file__).parent.parent.parent.parent
    raw_resumes_dir = base_dir / "data" / "raw" / "resumes"
    
    # Custom stopwords for resumes
    custom_stopwords = [
        'resume', 'cv', 'curriculum', 'vitae', 'page', 'of', 'the', 'and', 'a', 'an',
        'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'up', 'about', 'into',
        'over', 'after', 'email', 'phone', 'address', 'contact', 'information'
    ]
    
    if raw_resumes_dir.exists():
        results = preprocess_resume_directory(raw_resumes_dir, custom_stopwords)
        logger.info(f"Preprocessed {len(results)} resume files")
    else:
        logger.error(f"Directory not found: {raw_resumes_dir}")

if __name__ == "__main__":
    main() 