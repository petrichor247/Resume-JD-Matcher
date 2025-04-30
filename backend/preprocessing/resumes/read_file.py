import os
import PyPDF2
import docx
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResumeReader:
    """Class to handle reading and extracting text from resume files."""
    
    def __init__(self):
        """Initialize the ResumeReader."""
        pass
    
    def extract_text(self, file_path: str | Path) -> str:
        """
        Extract text from a resume file.
        
        Args:
            file_path (str | Path): Path to the resume file
            
        Returns:
            str: Extracted text from the resume
        """
        text = extract_text_from_file(file_path)
        if text is None:
            raise ValueError(f"Failed to extract text from {file_path}")
        return text
    
    def process_directory(self, directory_path: str | Path) -> dict:
        """
        Process all resume files in a directory.
        
        Args:
            directory_path (str | Path): Path to directory containing resumes
            
        Returns:
            dict: Mapping of original file paths to extracted text file paths
        """
        return process_resume_directory(directory_path)

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        logger.info(f"Successfully extracted text from PDF: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return None

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        logger.info(f"Successfully extracted text from DOCX: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        return None

def extract_text_from_txt(file_path):
    """Extract text from a TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        logger.info(f"Successfully extracted text from TXT: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from TXT {file_path}: {e}")
        return None

def extract_text_from_file(file_path):
    """Extract text from a file based on its extension."""
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        return None

def save_extracted_text(file_path, text):
    """Save extracted text to a file."""
    try:
        # Create extracted directory if it doesn't exist
        extracted_dir = Path(file_path).parent / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # Save text to file
        output_path = extracted_dir / f"{Path(file_path).stem}_extracted.txt"
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        
        logger.info(f"Saved extracted text to {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving extracted text: {e}")
        return None

def process_resume_directory(directory_path):
    """Process all resume files in a directory and extract text."""
    directory_path = Path(directory_path)
    results = {}
    
    for file_path in directory_path.glob("*"):
        if file_path.is_file():
            text = extract_text_from_file(file_path)
            if text:
                # Save extracted text to file
                saved_path = save_extracted_text(file_path, text)
                if saved_path:
                    results[str(file_path)] = saved_path
    
    return results

def main():
    """Main function to test file reading."""
    # Example usage
    base_dir = Path(__file__).parent.parent.parent.parent
    raw_resumes_dir = base_dir / "data" / "raw" / "resumes"
    
    reader = ResumeReader()
    if raw_resumes_dir.exists():
        results = reader.process_directory(raw_resumes_dir)
        logger.info(f"Processed {len(results)} resume files")
    else:
        logger.error(f"Directory not found: {raw_resumes_dir}")

if __name__ == "__main__":
    main() 