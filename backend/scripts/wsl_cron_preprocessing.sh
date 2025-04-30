#!/bin/bash

# WSL Cron Job Script for Resume-JD Search Engine Preprocessing
# This script is designed to run in WSL environment

# Set up logging
LOG_DIR="/mnt/d/IIT Madras/Academics/Senior year/42/mlops/ai_app/Resume-JD-Search-Engine/backend/logs"
mkdir -p "$LOG_DIR"

# Get current date for log file
DATE=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/preprocessing_${DATE}.log"

# Log start of execution
echo "=== Preprocessing job started at $(date) ===" >> "$LOG_FILE"

# Set the project directory (adjust this path to match your WSL path)
PROJECT_DIR="/mnt/d/IIT Madras/Academics/Senior year/42/mlops/ai_app/Resume-JD-Search-Engine/backend"
cd "$PROJECT_DIR" || { echo "Failed to change to project directory" >> "$LOG_FILE"; exit 1; }

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..." >> "$LOG_FILE"
    source venv/bin/activate
else
    echo "Virtual environment not found. Using system Python." >> "$LOG_FILE"
fi

# Check Python version
python --version >> "$LOG_FILE" 2>&1

# Run the resume preprocessing pipeline
echo "Starting resume preprocessing pipeline at $(date)" >> "$LOG_FILE"
python preprocessing/orchestrate.py --resumes >> "$LOG_FILE" 2>&1
RESUME_RESULT=$?

if [ $RESUME_RESULT -eq 0 ]; then
    echo "Resume preprocessing completed successfully at $(date)" >> "$LOG_FILE"
else
    echo "Resume preprocessing failed with exit code $RESUME_RESULT at $(date)" >> "$LOG_FILE"
fi

# Run the job description preprocessing pipeline
echo "Starting job description preprocessing pipeline at $(date)" >> "$LOG_FILE"
python preprocessing/orchestrate.py --jds \
    --search-queries "Data Scientist" "Machine Learning Engineer" "Software Engineer" \
    --location "United States" \
    --num-pages 2 >> "$LOG_FILE" 2>&1
JD_RESULT=$?

if [ $JD_RESULT -eq 0 ]; then
    echo "Job description preprocessing completed successfully at $(date)" >> "$LOG_FILE"
else
    echo "Job description preprocessing failed with exit code $JD_RESULT at $(date)" >> "$LOG_FILE"
fi

# Deactivate virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
    echo "Deactivated virtual environment" >> "$LOG_FILE"
fi

# Log end of execution
echo "=== Preprocessing job completed at $(date) ===" >> "$LOG_FILE"
echo "Resume pipeline result: $RESUME_RESULT, JD pipeline result: $JD_RESULT" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Return success if both pipelines completed successfully
if [ $RESUME_RESULT -eq 0 ] && [ $JD_RESULT -eq 0 ]; then
    exit 0
else
    exit 1
fi 