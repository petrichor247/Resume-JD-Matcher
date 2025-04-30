# Set the working directory to the project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptPath "..")

# Activate virtual environment if it exists
if (Test-Path "venv") {
    & .\venv\Scripts\Activate.ps1
}

# Set up logging directory
New-Item -ItemType Directory -Force -Path "logs"

# Get current date for log file
$date = Get-Date -Format "yyyyMMdd"

# Run the orchestrator with specific parameters
# Process resumes from directory
python preprocessing/orchestrate.py --resumes | Tee-Object -FilePath "logs/preprocessing_${date}.log" -Append

# Process job descriptions for specific roles
python preprocessing/orchestrate.py --jds `
    --search-queries "Data Scientist" "Machine Learning Engineer" "Software Engineer" `
    --location "United States" `
    --num-pages 2 | Tee-Object -FilePath "logs/preprocessing_${date}.log" -Append

# Deactivate virtual environment if it was activated
if (Test-Path "venv") {
    deactivate
} 