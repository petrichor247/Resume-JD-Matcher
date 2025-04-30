"""
Job Description preprocessing package.

This package provides functionality for preprocessing job descriptions:
1. Scraping job descriptions from LinkedIn
2. Text preprocessing (cleaning, normalization)
3. TF-IDF vectorization for semantic representation
"""

from .scraper import LinkedInScraper
from .preprocess import JDPreprocessor
from .vectorize import JDVectorizer

__all__ = [
    'LinkedInScraper',
    'JDPreprocessor',
    'JDVectorizer'
] 