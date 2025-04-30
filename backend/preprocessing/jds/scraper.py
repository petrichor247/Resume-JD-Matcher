import os
import time
import json
import csv
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UTC=__import__("datetime").timezone.utc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinkedInScraper:
    def __init__(self, output_dir: Optional[str] = None, proxies: Optional[List[str]] = None):
        """
        Initialize the LinkedIn scraper.
        
        Args:
            output_dir (str, optional): Directory to save scraped job descriptions
            proxies (List[str], optional): List of proxy URLs to use
        """
        self.base_url = "https://www.linkedin.com/jobs/search"
        self.ua = UserAgent()
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent.parent.parent / "data" / "raw" / "jobs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different file formats
        self.json_dir = self.output_dir / "json"
        self.csv_dir = self.output_dir / "csv"
        self.json_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)
        
        # Initialize session with retry strategy
        self.session = self._create_session()
        
        # Store proxies
        self.proxies = proxies or []
        self.current_proxy_index = 0
        
        # Rate limiting parameters
        self.min_delay = 5  # Minimum delay between requests
        self.max_delay = 15  # Maximum delay between requests
        self.current_delay = self.min_delay
        
        # Default search parameters for India
        self.default_params = {
            "location": "India",
            "f_TPR": "r86400",  # Last 24 hours
            "sortBy": "DD",     # Sort by date
            "f_E": "2,3,4,5",   # Experience levels: Entry, Associate, Mid-Senior, Director
            "f_JT": "F,T,C,P",  # Job types: Full-time, Part-time, Contract, Internship
            "keywords": "",     # Empty keywords to get all jobs
        }
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,  # number of retries
            backoff_factor=1,  # wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get random headers for requests."""
        return {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
            "TE": "Trailers",
        }
    
    def _get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next proxy from the list."""
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        
        return {
            "http": proxy,
            "https": proxy
        }
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[BeautifulSoup]:
        """
        Make a request to LinkedIn and return BeautifulSoup object.
        
        Args:
            url (str): URL to request
            params (dict, optional): Query parameters
            
        Returns:
            BeautifulSoup: Parsed HTML content
        """
        try:
            # Add random delay
            time.sleep(self.current_delay + random.uniform(0, 2))
            
            # Get proxy if available
            proxies = self._get_next_proxy()
            
            # Log the request details
            logger.info(f"Making request to: {url}")
            logger.info(f"With parameters: {params}")
            if proxies:
                logger.info(f"Using proxy: {proxies}")
            
            # Make request
            response = self.session.get(
                url,
                params=params,
                headers=self._get_headers(),
                proxies=proxies,
                timeout=30
            )
            
            # Log response status
            logger.info(f"Response status code: {response.status_code}")
            
            # Check for rate limiting
            if response.status_code == 429:
                logger.warning("Rate limited by LinkedIn. Increasing delay...")
                self.current_delay = min(self.current_delay * 2, self.max_delay)
                time.sleep(self.current_delay)
                return self._make_request(url, params)  # Retry request
            
            response.raise_for_status()
            
            # Log response content length
            logger.info(f"Response content length: {len(response.text)}")
            
            return BeautifulSoup(response.text, "html.parser")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {e}")
            if "429" in str(e):
                self.current_delay = min(self.current_delay * 2, self.max_delay)
                time.sleep(self.current_delay)
                return self._make_request(url, params)  # Retry request
            return None
    
    def _extract_job_data(self, job_card) -> Optional[Dict]:
        """
        Extract job data from a job card.
        
        Args:
            job_card: BeautifulSoup object representing a job card
            
        Returns:
            dict: Extracted job data
        """
        try:
            # Extract job title
            title_elem = job_card.find("h3", class_="base-search-card__title")
            title = title_elem.text.strip() if title_elem else "Unknown Title"
            
            # Extract company name
            company_elem = job_card.find("h4", class_="base-search-card__subtitle")
            company = company_elem.text.strip() if company_elem else "Unknown Company"
            
            # Extract location
            location_elem = job_card.find("span", class_="job-search-card__location")
            location = location_elem.text.strip() if location_elem else "Unknown Location"
            
            # Extract job URL
            link_elem = job_card.find("a", class_="base-card__full-link")
            job_url = link_elem.get("href") if link_elem else None
            
            if not job_url:
                return None
            
            # Get job description
            job_soup = self._make_request(job_url)
            if not job_soup:
                return None
            
            description_elem = job_soup.find("div", class_="show-more-less-html__markup")
            description = description_elem.text.strip() if description_elem else ""
            
            # Extract additional metadata
            metadata = self._extract_job_metadata(job_soup)
            
            # Generate a unique ID
            job_id = f"jd_{int(time.time() * 1000)}_{hash(title + company + location) % 10000:04d}"
            
            return {
                "id": job_id,
                "title": title,
                "company": company,
                "location": location,
                "url": job_url,
                "description": description,
                "timestamp": datetime.now(UTC).isoformat(),
                **metadata
            }
        except Exception as e:
            logger.error(f"Error extracting job data: {e}")
            return None
    
    def _extract_job_metadata(self, job_soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract additional job metadata from the job page.
        
        Args:
            job_soup (BeautifulSoup): Parsed job page HTML
            
        Returns:
            dict: Additional job metadata
        """
        metadata = {}
        
        try:
            # Extract job type
            job_type_elem = job_soup.find("span", class_="job-criteria-item__text")
            if job_type_elem:
                metadata["job_type"] = job_type_elem.text.strip()
            
            # Extract experience level
            experience_elem = job_soup.find("span", class_="job-criteria-item__text", string=lambda x: x and "experience" in x.lower())
            if experience_elem:
                metadata["experience_level"] = experience_elem.text.strip()
            
            # Extract industry
            industry_elem = job_soup.find("span", class_="job-criteria-item__text", string=lambda x: x and "industry" in x.lower())
            if industry_elem:
                metadata["industry"] = industry_elem.text.strip()
            
            # Extract employment type
            employment_type_elem = job_soup.find("span", class_="job-criteria-item__text", string=lambda x: x and "employment" in x.lower())
            if employment_type_elem:
                metadata["employment_type"] = employment_type_elem.text.strip()
            
            # Extract posted date
            posted_date_elem = job_soup.find("span", class_="posted-time-ago__text")
            if posted_date_elem:
                metadata["posted_date"] = posted_date_elem.text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting job metadata: {e}")
        
        return metadata
    
    def _save_job_to_json(self, job_data: Dict) -> bool:
        """
        Save job data to a JSON file.
        
        Args:
            job_data (dict): Job data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = self.json_dir / f"{job_data['id']}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(job_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving job data to JSON: {e}")
            return False
    
    def _save_jobs_to_csv(self, jobs: List[Dict]) -> bool:
        """
        Save job data to a CSV file.
        
        Args:
            jobs (list): List of job data dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not jobs:
                return False
            
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.csv_dir / f"linkedin_jobs_{timestamp}.csv"
            
            # Get all possible fieldnames from the job data
            fieldnames = set()
            for job in jobs:
                fieldnames.update(job.keys())
            fieldnames = sorted(list(fieldnames))
            
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(jobs)
            
            return True
        except Exception as e:
            logger.error(f"Error saving job data to CSV: {e}")
            return False
    
    def scrape_jobs(self, search_params: Optional[Dict[str, Any]] = None, num_pages: int = 1) -> Dict[str, Dict]:
        """
        Scrape job descriptions from LinkedIn.
        
        Args:
            search_params (dict, optional): Search parameters
            num_pages (int): Number of pages to scrape
            
        Returns:
            dict: Dictionary of scraped job descriptions
        """
        try:
            # Merge default params with search params
            params = self.default_params.copy()
            if search_params:
                params.update(search_params)
            
            # Ensure location is set to India
            params["location"] = "India"
            
            # Log the search parameters
            logger.info(f"Starting job search with parameters: {params}")
            
            # Initialize results
            jobs = {}
            page = 0
            
            while page < num_pages:
                # Update page parameter
                params["start"] = page * 25  # LinkedIn shows 25 jobs per page
                
                # Construct URL
                url = f"{self.base_url}?{requests.compat.urlencode(params)}"
                logger.info(f"Scraping page {page + 1} of {num_pages}")
                logger.info(f"URL: {url}")
                
                # Get page content
                soup = self._make_request(url)
                if not soup:
                    logger.error(f"Failed to get page {page + 1}")
                    break
                
                # Find job cards
                job_cards = soup.find_all("div", class_="base-card")
                logger.info(f"Found {len(job_cards)} job cards on page {page + 1}")
                
                if not job_cards:
                    logger.warning(f"No job cards found on page {page + 1}")
                    # Log the HTML content for debugging
                    logger.debug(f"Page HTML content: {soup.prettify()}")
                    break
                
                # Process each job card
                jobcount = 0
                for card in job_cards:
                    if (jobcount > 10):
                        break
                    job_data = self._extract_job_data(card)
                    if job_data:
                        jobs[job_data["id"]] = job_data
                        # Save to JSON
                        self._save_job_to_json(job_data)
                        logger.info(f"Successfully processed job: {job_data['id']}")
                    else:
                        logger.warning("Failed to extract job data from card")
                    jobcount += 1
                # Save batch to CSV
                if jobs:
                    self._save_jobs_to_csv(list(jobs.values()))
                
                page += 1
                
                # Add delay between pages
                time.sleep(self.current_delay + random.uniform(0, 2))
            
            logger.info(f"Scraped {len(jobs)} jobs")
            return jobs
            
        except Exception as e:
            logger.error(f"Error scraping jobs: {e}")
            return {}

def main():
    """Main function to test the scraper."""
    # Example usage with proxies (optional)
    proxies = [
        # Add your proxy URLs here
        # "http://proxy1.example.com:8080",
        # "http://proxy2.example.com:8080"
    ]
    
    scraper = LinkedInScraper(proxies=proxies)
    
    # Example search parameters
    search_params = {
        "keywords": "Data Scientist OR Machine Learning Engineer",
        "location": "San Francisco, CA",
        "f_TPR": "r86400",  # Last 24 hours
        "sortBy": "DD",     # Sort by date
        "f_E": "2,3",       # Entry and Associate level
        "f_JT": "F",        # Full-time only
    }
    
    jobs = scraper.scrape_jobs(search_params=search_params, num_pages=1)
    
    logger.info(f"Scraped {len(jobs)} job descriptions")
    logger.info(f"Job descriptions saved to {scraper.output_dir}")

if __name__ == "__main__":
    main()  