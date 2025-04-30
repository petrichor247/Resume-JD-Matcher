import os
import shutil
import kagglehub
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).parent.parent.parent.parent
    raw_resumes_dir = base_dir / "data" / "raw" / "resumes"
    
    # Create directories if they don't exist
    raw_resumes_dir.mkdir(parents=True, exist_ok=True)
    
    return raw_resumes_dir

def fetch_kaggle_resumes(raw_resumes_dir):
    """Download resumes from Kaggle dataset."""
    try:
        logger.info("Downloading resumes from Kaggle dataset...")
        # Download latest version
        path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
        logger.info(f"Path to dataset files: {path}")
        
        # Copy files to our raw resumes directory
        for file in Path(path).glob("**/*"):
            if file.is_file():
                # Determine the destination path
                dest_path = raw_resumes_dir / file.name
                
                # Copy the file
                shutil.copy2(file, dest_path)
                logger.info(f"Copied {file.name} to {dest_path}")
        
        logger.info("Successfully downloaded and copied Kaggle resumes")
        return True
    except Exception as e:
        logger.error(f"Error downloading Kaggle resumes: {e}")
        return False

def save_user_resume(file_path, raw_resumes_dir):
    """Save a user-uploaded resume to the raw resumes directory."""
    try:
        # Get the filename from the path
        filename = os.path.basename(file_path)
        
        # Determine the destination path
        dest_path = raw_resumes_dir / filename
        
        # Copy the file
        shutil.copy2(file_path, dest_path)
        logger.info(f"Saved user resume {filename} to {dest_path}")
        
        return str(dest_path)
    except Exception as e:
        logger.error(f"Error saving user resume: {e}")
        return None

def main():
    """Main function to fetch resumes."""
    raw_resumes_dir = setup_directories()
    fetch_kaggle_resumes(raw_resumes_dir)
    logger.info(f"Resume fetching completed. Resumes stored in {raw_resumes_dir}")

if __name__ == "__main__":
    main() 