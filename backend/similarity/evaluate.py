import os
import argparse
import logging
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from model import SiameseBiLSTMAttention, SimilarityDataset
from train import load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(
    model: SiameseBiLSTMAttention,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    experiment_name: str = "resume-jd-similarity-evaluation"
) -> Dict[str, float]:
    """
    Evaluate the model on test data with MLflow tracking.
    
    Args:
        model (SiameseBiLSTMAttention): Trained model
        test_loader (DataLoader): Test data loader
        device (str): Device to use for computation
        experiment_name (str): Name of the MLflow experiment
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Initialize metrics
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Evaluation loop
    with mlflow.start_run():
        with torch.no_grad():
            for x1, x2, labels in test_loader:
                # Forward pass
                outputs, _ = model(x1, x2)
                
                # Get predictions
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                # Store results
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds),
            "recall": recall_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds),
            "roc_auc": roc_auc_score(all_labels, all_probs)
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save and log confusion matrix
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Create ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        
        # Save and log ROC curve
        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
    
    return metrics

def main():
    """Main function to evaluate the model with MLflow tracking."""
    parser = argparse.ArgumentParser(description="Evaluate Siamese BiLSTM model for resume-JD similarity")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--resume_dir", type=str, required=True, help="Directory containing preprocessed resume texts")
    parser.add_argument("--jd_dir", type=str, required=True, help="Directory containing preprocessed job description texts")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--experiment_name", type=str, default="resume-jd-similarity-evaluation", help="MLflow experiment name")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    resume_texts, jd_texts = load_data(args.resume_dir, args.jd_dir)
    
    if not resume_texts or not jd_texts:
        logger.error("No data found in the specified directories")
        return
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset
    dataset = SimilarityDataset(
        resume_texts=resume_texts,
        jd_texts=jd_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        device=device
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Load model
    try:
        model = SiameseBiLSTMAttention.load(args.model_path)
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        experiment_name=args.experiment_name
    )
    
    # Save metrics
    metrics_path = Path("evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main() 