#!/bin/bash

# Set the working directory to the project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set up logging directory
mkdir -p logs

# Get current date for log file
DATE=$(date +%Y%m%d)

# Run the orchestrator with specific parameters
# Process resumes from directory
python preprocessing/orchestrate.py --resumes >> "logs/preprocessing_${DATE}.log" 2>&1

# Process job descriptions for specific roles
python preprocessing/orchestrate.py --jds \
    --search-queries "Data Scientist" "Machine Learning Engineer" "Software Engineer" \
    --location "United States" \
    --num-pages 2 >> "logs/preprocessing_${DATE}.log" 2>&1

# Deactivate virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi 