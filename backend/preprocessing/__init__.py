"""
Preprocessing package for the Resume-JD Matcher application.

This package provides functionality for preprocessing both resumes and job descriptions,
including fetching, reading, preprocessing, and vectorizing text data.
"""

from . import resumes
from . import jds

__all__ = [
    'resumes',
    'jds'
] 