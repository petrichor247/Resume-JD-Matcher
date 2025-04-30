import os
import kagglehub
import shutil
from pathlib import Path
import zipfile
import json
import PyPDF2
from data_processor import DataProcessor

class DataDownloader:
    def __init__(self, data_dir="data"):
        """Initialize the DataDownloader.
        
        Args:
            data_dir (str): Base directory for data storage
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.embeddings_dir = os.path.join(data_dir, "embeddings")
        self.resumes_dir = os.path.join(data_dir, "resumes")
        self.user_resumes_dir = os.path.join(self.resumes_dir, "user_uploads")
        
        
        # Create necessary directories
        for directory in [self.raw_dir, self.processed_dir, self.embeddings_dir, 
                         self.resumes_dir, self.user_resumes_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def get_processed_resumes(self):
        """Get the set of already processed resume filenames.
        
        Returns:
            set: Set of processed resume filenames
        """
        processed_files = set()
        for file in Path(self.processed_dir).glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'filename' in data:
                    processed_files.add(data['filename'])
        return processed_files
    
    def process_existing_resumes(self, data_processor=None):
        """Process existing resumes from the resumes directory."""
        if data_processor is None:
            data_processor = DataProcessor(data_dir=self.data_dir)
        
        print("Processing existing resumes...")
        
        # Get already processed resumes
        processed_resumes = self.get_processed_resumes()
        
        # Find all PDF files in the resumes directory
        pdf_files = []
        for root, _, files in os.walk(self.resumes_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        print(f"Found {len(pdf_files)} PDF files in the resumes directory")
        
        # Process each PDF file
        processed_count = 0
        for pdf_path in pdf_files:
            if (processed_count > 200):
                break
            try:
                # Skip if already processed
                if os.path.basename(pdf_path) in processed_resumes:
                    print(f"Skipping already processed resume: {os.path.basename(pdf_path)}")
                    continue
                
                # Check if the file exists and is not empty
                if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
                    print(f"Skipping empty or non-existent file: {pdf_path}")
                    continue
                
                # Extract text from PDF
                text = self._extract_text_from_pdf(pdf_path)
                if not text:
                    print(f"No text extracted from {pdf_path}")
                    continue
                
                # Read the PDF file as binary
                with open(pdf_path, 'rb') as f:
                    pdf_content = f.read()
                
                # Save the resume text to the data processor
                resume_id = data_processor.save_resume(text, original_file=pdf_content)
                
                if resume_id:
                    processed_count += 1
                    print(f"Processed resume: {os.path.basename(pdf_path)} (ID: {resume_id})")
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
        
        print(f"Successfully processed {processed_count} new resumes")
        return processed_count
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return None
    
    def process_all_resumes(self):
        """Process all existing resumes."""
        return self.process_existing_resumes()
    
    def save_user_resume(self, file_content, filename):
        """Save a user-uploaded resume to the appropriate directory.
        
        Args:
            file_content (bytes): The binary content of the uploaded file
            filename (str): Original filename of the uploaded file
            
        Returns:
            str: Path to the saved file, or None if save failed
        """
        try:
            # Ensure filename is safe and has .pdf extension
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            if not safe_filename.lower().endswith('.pdf'):
                safe_filename += '.pdf'
            
            # Create a unique filename to prevent overwrites
            base_name = os.path.splitext(safe_filename)[0]
            counter = 1
            final_filename = safe_filename
            while os.path.exists(os.path.join(self.user_resumes_dir, final_filename)):
                final_filename = f"{base_name}_{counter}.pdf"
                counter += 1
            
            # Save the file
            file_path = os.path.join(self.user_resumes_dir, final_filename)
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            print(f"Saved user resume to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error saving user resume: {str(e)}")
            return None
    
    def process_user_resume(self, file_content, filename, data_processor=None):
        """Process a user-uploaded resume.
        
        Args:
            file_content (bytes): The binary content of the uploaded file
            filename (str): Original filename of the uploaded file
            data_processor (DataProcessor, optional): DataProcessor instance
            
        Returns:
            str: Resume ID if processing successful, None otherwise
        """
        try:
            # Save the file first
            file_path = self.save_user_resume(file_content, filename)
            if not file_path:
                return None
            
            # Initialize data processor if not provided
            if data_processor is None:
                data_processor = DataProcessor(data_dir=self.data_dir)
            
            # Extract text from PDF
            text = self._extract_text_from_pdf(file_path)
            if not text:
                print(f"No text extracted from {file_path}")
                return None
            
            # Save the resume text
            resume_id = data_processor.save_resume(text, original_file=file_content)
            
            if resume_id:
                print(f"Processed user resume: {filename} (ID: {resume_id})")
                return resume_id
            else:
                print(f"Failed to process user resume: {filename}")
                return None
                
        except Exception as e:
            print(f"Error processing user resume: {str(e)}")
            return None
    
    def get_all_resumes(self):
        """Get paths to all resumes (both from dataset and user uploads).
        
        Returns:
            list: List of paths to all PDF files
        """
        pdf_files = []
        for root, _, files in os.walk(self.resumes_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files 
