import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
from datetime import datetime
import json

from .resumes.read_file import ResumeReader
from .resumes.preprocess import ResumePreprocessor
from .resumes.vectorize import ResumeVectorizer
from .jds.scraper import LinkedInScraper
from .jds.preprocess import JDPreprocessor
from .jds.vectorize import JDVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories() -> Dict[str, Path]:
    """
    Create necessary directories for the preprocessing pipeline.
    
    Returns:
        Dict[str, Path]: Dictionary of directory paths
    """
    base_dir = Path("data")
    directories = {
        "logs": base_dir / "logs",
        "resumes": base_dir / "resumes" / "data" / "data",
        "processed": base_dir / "processed",
        "metadata": base_dir / "metadata"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return directories

def get_processed_resumes() -> set:
    """
    Get the set of already processed resume filenames.
    
    Returns:
        set: Set of processed resume filenames
    """
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        return set()
    
    processed_files = set()
    for file in processed_dir.glob("*.json"):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'filename' in data:
                processed_files.add(data['filename'])
    
    return processed_files

def run_resume_pipeline(resume_path: Optional[str] = None, resume_dir: Optional[str] = None) -> int:
    """
    Run the resume preprocessing pipeline.
    
    Args:
        resume_path (str, optional): Path to a single resume file
        resume_dir (str, optional): Directory containing resume files
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Setup directories
        dirs = setup_directories()
        
        # Initialize components
        reader = ResumeReader()
        preprocessor = ResumePreprocessor()
        vectorizer = ResumeVectorizer()
        
        # Get already processed resumes
        processed_resumes = get_processed_resumes()
        
        # Process resumes
        if resume_path:
            # Process single resume
            resume_file = Path(resume_path)
            if not resume_file.exists():
                raise FileNotFoundError(f"Resume file not found: {resume_path}")
            
            if resume_file.name in processed_resumes:
                logger.info(f"Resume already processed: {resume_file.name}")
                return 0
            
            # Read PDF content
            with open(resume_file, "rb") as f:
                pdf_content = f.read()
            
            # Extract text
            extracted_text = reader.extract_text(resume_file)
            
            # Preprocess text
            preprocessed_text = preprocessor.preprocess(extracted_text)
            
            # Vectorize text
            vector = vectorizer.vectorize(preprocessed_text)
            
            # Save processed data
            processed_data = {
                "filename": resume_file.name,
                "text": preprocessed_text,
                "vector": vector.tolist(),
                "processed_at": datetime.now().isoformat()
            }
            
            processed_file = dirs["processed"] / f"{resume_file.stem}.json"
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Processed resume: {resume_file.name}")
        
        elif resume_dir:
            # Process all resumes in directory
            resume_path = Path(resume_dir)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
            
            for resume_file in resume_path.glob("*.pdf"):
                try:
                    if resume_file.name in processed_resumes:
                        logger.info(f"Resume already processed: {resume_file.name}")
                        continue
                    
                    # Read PDF content
                    with open(resume_file, "rb") as f:
                        pdf_content = f.read()
                    
                    # Extract text
                    extracted_text = reader.extract_text(resume_file)
                    
                    # Preprocess text
                    preprocessed_text = preprocessor.preprocess(extracted_text)
                    
                    # Vectorize text
                    vector = vectorizer.vectorize(preprocessed_text)
                    
                    # Save processed data
                    processed_data = {
                        "filename": resume_file.name,
                        "text": preprocessed_text,
                        "vector": vector.tolist(),
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    processed_file = dirs["processed"] / f"{resume_file.stem}.json"
                    with open(processed_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Processed resume: {resume_file.name}")
                
                except Exception as e:
                    logger.error(f"Error processing resume {resume_file.name}: {e}")
                    continue
        
        else:
            raise ValueError("Either resume_path or resume_dir must be provided")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in resume pipeline: {e}")
        return 1

def run_jd_pipeline(search_queries: List[str], location: Optional[str] = None, num_pages: int = 1) -> int:
    """
    Run the job description preprocessing pipeline.
    
    Args:
        search_queries (List[str]): List of search queries for job descriptions
        location (str, optional): Location to filter job descriptions
        num_pages (int): Number of pages to scrape per query
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Setup directories
        dirs = setup_directories()
        
        # Initialize components
        scraper = LinkedInScraper()
        preprocessor = JDPreprocessor()
        vectorizer = JDVectorizer()
        
        # Process job descriptions
        for query in search_queries:
            try:
                # Scrape job descriptions
                search_params = {
                    "keywords": query,
                    "location": location
                }
                jds = scraper.scrape_jobs(search_params, num_pages)
                
                for jd in jds:
                    try:
                        # Preprocess text
                        preprocessed_text = preprocessor.preprocess(jd["description"])
                        
                        # Vectorize text
                        vector = vectorizer.vectorize(preprocessed_text)
                        
                        # Save processed data
                        processed_data = {
                            "filename": f"{jd['id']}.txt",
                            "text": preprocessed_text,
                            "metadata": jd,
                            "job_title": jd.get("title"),
                            "company": jd.get("company"),
                            "location": jd.get("location"),
                            "vector": vector.tolist(),
                            "processed_at": datetime.now().isoformat()
                        }
                        
                        processed_file = dirs["processed"] / f"{jd['id']}.json"
                        with open(processed_file, 'w', encoding='utf-8') as f:
                            json.dump(processed_data, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"Processed job description: {jd['id']}")
                    
                    except Exception as e:
                        logger.error(f"Error processing job description {jd['id']}: {e}")
                        continue
                
                # Add delay between queries to avoid rate limiting
                time.sleep(5)
            
            except Exception as e:
                logger.error(f"Error processing search query '{query}': {e}")
                continue
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in job description pipeline: {e}")
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run preprocessing pipelines")
    parser.add_argument("--resumes", help="Directory containing resume files")
    parser.add_argument("--user-resume", help="Path to a single resume file")
    parser.add_argument("--jds", action="store_true", help="Run job description pipeline")
    parser.add_argument("--search-queries", nargs="+", help="Search queries for job descriptions")
    parser.add_argument("--location", help="Location to filter job descriptions")
    parser.add_argument("--num-pages", type=int, default=1, help="Number of pages to scrape per query")
    
    args = parser.parse_args()
    
    # Run resume pipeline if specified
    if args.user_resume or args.resumes:
        logger.info("Starting resume preprocessing pipeline...")
        exit_code = run_resume_pipeline(args.user_resume, args.resumes)
        if exit_code != 0:
            logger.error("Resume pipeline failed")
        else:
            logger.info("Resume pipeline completed successfully")
    
    # Run JD pipeline if specified or if no specific pipeline is requested
    if args.jds or (not args.user_resume and not args.resumes):
        logger.info("Starting job description preprocessing pipeline...")
        # Use empty search queries to get all jobs in India
        exit_code = run_jd_pipeline(
            search_queries=[],  # Empty list to get all jobs
            location="India",   # Hardcoded to India
            num_pages=1         # Hardcoded to 1 page
        )
        if exit_code != 0:
            logger.error("Job description pipeline failed")
        else:
            logger.info("Job description pipeline completed successfully")
    
    logger.info("All preprocessing pipelines completed") 