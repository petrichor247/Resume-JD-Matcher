import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime

from model import SiameseBiLSTMAttention
from train import load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimilarityPredictor:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5
    ):
        """
        Initialize the similarity predictor.
        
        Args:
            model_path (str): Path to the trained model
            tokenizer_name (str): Name of the tokenizer to use
            max_length (int): Maximum sequence length
            device (str): Device to use for computation
            threshold (float): Threshold for similarity score
        """
        # Load model
        self.model = SiameseBiLSTMAttention.load(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.max_length = max_length
        self.device = device
        self.threshold = threshold
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {device}")
    
    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess text for inference.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            torch.Tensor: Preprocessed text tensor
        """
        # Tokenize text
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to tensor
        tensor = tokens['input_ids'].squeeze(0).to(self.device)
        
        return tensor
    
    def predict_similarity(self, resume_text: str, jd_text: str) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Predict similarity between resume and job description.
        
        Args:
            resume_text (str): Resume text
            jd_text (str): Job description text
            
        Returns:
            Tuple[float, Dict[str, torch.Tensor]]: Similarity score and attention weights
        """
        # Preprocess texts
        resume_tensor = self.preprocess_text(resume_text)
        jd_tensor = self.preprocess_text(jd_text)
        
        # Add batch dimension
        resume_tensor = resume_tensor.unsqueeze(0)
        jd_tensor = jd_tensor.unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            similarity_score, attention_weights = self.model(resume_tensor, jd_tensor)
        
        # Convert to float
        similarity_score = similarity_score.item()
        
        return similarity_score, attention_weights
    
    def find_matching_jds(
        self,
        resume_text: str,
        jd_texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Find matching job descriptions for a resume.
        
        Args:
            resume_text (str): Resume text
            jd_texts (List[str]): List of job description texts
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of matching job descriptions with similarity scores
        """
        # Preprocess resume
        resume_tensor = self.preprocess_text(resume_text)
        resume_tensor = resume_tensor.unsqueeze(0)
        
        # Initialize results
        results = []
        
        # Predict similarity for each JD
        for i, jd_text in enumerate(jd_texts):
            # Preprocess JD
            jd_tensor = self.preprocess_text(jd_text)
            jd_tensor = jd_tensor.unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                similarity_score, _ = self.model(resume_tensor, jd_tensor)
            
            # Convert to float
            similarity_score = similarity_score.item()
            
            # Add to results
            results.append({
                "jd_index": i,
                "jd_text": jd_text,
                "similarity_score": similarity_score,
                "is_match": similarity_score >= self.threshold
            })
        
        # Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Return top k
        return results[:top_k]
    
    def find_matching_resumes(
        self,
        jd_text: str,
        resume_texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Find matching resumes for a job description.
        
        Args:
            jd_text (str): Job description text
            resume_texts (List[str]): List of resume texts
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of matching resumes with similarity scores
        """
        # Preprocess JD
        jd_tensor = self.preprocess_text(jd_text)
        jd_tensor = jd_tensor.unsqueeze(0)
        
        # Initialize results
        results = []
        
        # Predict similarity for each resume
        for i, resume_text in enumerate(resume_texts):
            # Preprocess resume
            resume_tensor = self.preprocess_text(resume_text)
            resume_tensor = resume_tensor.unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                similarity_score, _ = self.model(resume_tensor, jd_tensor)
            
            # Convert to float
            similarity_score = similarity_score.item()
            
            # Add to results
            results.append({
                "resume_index": i,
                "resume_text": resume_text,
                "similarity_score": similarity_score,
                "is_match": similarity_score >= self.threshold
            })
        
        # Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Return top k
        return results[:top_k]

def main():
    """Main function to run inference with the trained model."""
    parser = argparse.ArgumentParser(description="Run inference with Siamese BiLSTM model for resume-JD similarity")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--resume_dir", type=str, required=True, help="Directory containing preprocessed resume texts")
    parser.add_argument("--jd_dir", type=str, required=True, help="Directory containing preprocessed job description texts")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches to return")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for similarity score")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    resume_texts, jd_texts = load_data(args.resume_dir, args.jd_dir)
    
    if not resume_texts or not jd_texts:
        logger.error("No data found in the specified directories")
        return
    
    # Initialize predictor
    predictor = SimilarityPredictor(
        model_path=args.model_path,
        threshold=args.threshold
    )
    
    # Find matches for each resume
    resume_matches = []
    for i, resume_text in enumerate(resume_texts):
        matches = predictor.find_matching_jds(
            resume_text=resume_text,
            jd_texts=jd_texts,
            top_k=args.top_k
        )
        
        resume_matches.append({
            "resume_index": i,
            "resume_text": resume_text,
            "matches": matches
        })
    
    # Find matches for each JD
    jd_matches = []
    for i, jd_text in enumerate(jd_texts):
        matches = predictor.find_matching_resumes(
            jd_text=jd_text,
            resume_texts=resume_texts,
            top_k=args.top_k
        )
        
        jd_matches.append({
            "jd_index": i,
            "jd_text": jd_text,
            "matches": matches
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    resume_results_path = output_dir / f"resume_matches_{timestamp}.json"
    with open(resume_results_path, "w") as f:
        json.dump(resume_matches, f, indent=2)
    
    jd_results_path = output_dir / f"jd_matches_{timestamp}.json"
    with open(jd_results_path, "w") as f:
        json.dump(jd_matches, f, indent=2)
    
    logger.info(f"Resume matches saved to {resume_results_path}")
    logger.info(f"JD matches saved to {jd_results_path}")
    logger.info("Inference completed successfully")

if __name__ == "__main__":
    main() 