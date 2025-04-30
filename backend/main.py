from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import tempfile
import os
from pathlib import Path
import json
from preprocessing.orchestrate import run_resume_pipeline, run_jd_pipeline
from preprocessing.data_downloader import DataDownloader
from preprocessing.jds.preprocess import preprocess_jds
from similarity.model import SimilarityModel
from similarity.inference import SimilarityPredictor

app = FastAPI(title="Resume-JD Matcher API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_downloader = DataDownloader()
pre_process_jds = preprocess_jds()
model = SimilarityModel()
predictor = SimilarityPredictor(model)

class JobDescription(BaseModel):
    title: str
    description: str
    requirements: List[str]
    company: Optional[str] = None

class Resume(BaseModel):
    content: str
    skills: List[str]
    experience: List[dict]

@app.get("/")
async def root():
    return {"message": "Welcome to Resume-JD Matcher API"}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the user-uploaded resume
            resume_id = data_downloader.process_user_resume(temp_file.name, file.filename)
            
            if not resume_id:
                raise HTTPException(status_code=500, detail="Failed to process resume")
            
            # Get the processed resume data
            resume_path = data_downloader.get_resume_path(resume_id)
            resume_text = data_downloader.get_resume_text(resume_id)
            
            return {
                "message": "Resume processed successfully",
                "filename": file.filename,
                "content": resume_text,
                "resume_id": resume_id
            }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        # Clean up the temporary file
        if 'temp_file' in locals():
            os.unlink(temp_file.name)

@app.post("/match")
async def match_resume_jd(resume_id: int):
    try:
        # Get the resume path
        resume_path = data_downloader.get_resume_path(resume_id)
        if not resume_path:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Get the resume text
        resume_text = data_downloader.get_resume_text(resume_id)
        
        # Get all processed JDs
        jd_dir = Path("data/processed")
        matches = []
        
        for jd_file in jd_dir.glob("*.json"):
            try:
                with open(jd_file, 'r', encoding='utf-8') as f:
                    jd_data = json.load(f)
                    
                    # Get similarity score
                    similarity_score = predictor.predict_similarity(
                        resume_text=resume_text,
                        jd_text=jd_data["text"]
                    )
                    
                    matches.append({
                        "jd_id": jd_data["id"],
                        "title": jd_data["title"],
                        "company": jd_data["company"],
                        "location": jd_data["location"],
                        "similarity_score": float(similarity_score),
                        "description": jd_data["description"]
                    })
            except Exception as e:
                print(f"Error processing JD {jd_file}: {e}")
                continue
        
        # Sort matches by similarity score
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "resume": {
                "id": resume_id,
                "text": resume_text,
                "path": resume_path
            },
            "matches": matches
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 