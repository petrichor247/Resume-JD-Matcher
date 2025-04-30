import os
import json
import logging
from pathlib import Path
from preprocessing.orchestrate import run_jd_pipeline
from preprocessing.jds.scraper import LinkedInScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/jd_preprocessing.log'),
    ]
)
logger = logging.getLogger(__name__)

def preprocess_jds():
    """
    Preprocess job descriptions by scraping from LinkedIn and saving to the processed directory.
    This is a standalone preprocessing step, similar to resume preprocessing.
    """
    try:
        # Create logs directory if it doesn't exist
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting JD preprocessing")
        
        # Test the scraper directly first
        logger.info("Testing LinkedIn scraper...")
        scraper = LinkedInScraper()
        test_jobs = scraper.scrape_jobs(
            search_params={"keywords": "", "location": "India"},
            num_pages=1
        )
        
        if not test_jobs:
            logger.error("LinkedIn scraper test failed - no jobs were scraped")
            return False
        
        logger.info(f"LinkedIn scraper test successful - scraped {len(test_jobs)} jobs")
        
        # Run the JD pipeline with hardcoded values
        logger.info("Running JD pipeline...")
        exit_code = run_jd_pipeline(
            search_queries=[],  # Empty list to get all jobs
            location="India",
            num_pages=1
        )
        
        if exit_code != 0:
            logger.error("Failed to scrape job descriptions")
            return False
        
        # Get the scraped JDs from the data directory
        jd_dir = Path("data/raw/jobs")
        if not jd_dir.exists():
            logger.error(f"JD directory not found: {jd_dir}")
            return False
        
        logger.info(f"Found JD directory: {jd_dir}")
        
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and save each JD
        jd_files = list(jd_dir.glob("*.json"))
        logger.info(f"Found {len(jd_files)} JD files to process")
        
        for jd_file in jd_files:
            try:
                with open(jd_file, 'r', encoding='utf-8') as f:
                    jd_data = json.load(f)
                    # Process and save the JD
                    processed_jd = {
                        "id": jd_data.get("id", ""),
                        "title": jd_data.get("title", ""),
                        "company": jd_data.get("company", ""),
                        #"location": jd_data.get("location", ""),
                        "description": jd_data.get("description", "")
                        #"metadata": jd_data.get("metadata", {})
                    }
                    
                    # Save processed JD to the processed directory
                    processed_file = processed_dir / f"{jd_data['id']}.json"
                    with open(processed_file, 'w', encoding='utf-8') as pf:
                        json.dump(processed_jd, pf, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Processed JD: {jd_data['id']}")
            except Exception as e:
                logger.error(f"Error processing JD file {jd_file}: {e}")
                continue
        
        logger.info("JD preprocessing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in JD preprocessing: {e}")
        return False

if __name__ == "__main__":
    preprocess_jds() 
