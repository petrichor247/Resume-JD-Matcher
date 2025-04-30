import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import pickle
from datetime import datetime

from .database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(db_path: str = "data/preprocessed.db") -> Tuple[List[str], List[str], List[bool]]:
    """
    Load resume and job description texts from the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file
        
    Returns:
        Tuple[List[str], List[str], List[bool]]: Lists of resume texts, job description texts, and labels
    """
    logger.info("Loading data from database...")
    
    with DatabaseManager(db_path) as db:
        # Get all resumes and job descriptions
        resumes = db.get_all_resumes()
        job_descriptions = db.get_all_job_descriptions()
        
        # Get similarity pairs (labeled data)
        similarity_pairs = db.get_similarity_pairs()
        
        # Extract texts and labels
        resume_texts = []
        jd_texts = []
        labels = []
        
        for pair in similarity_pairs:
            resume = db.get_resume_by_id(pair["resume_id"])
            jd = db.get_job_description_by_id(pair["jd_id"])
            
            if resume and jd and pair["is_match"] is not None:
                resume_texts.append(resume["text"])
                jd_texts.append(jd["text"])
                labels.append(pair["is_match"])
        
        logger.info(f"Loaded {len(resume_texts)} labeled pairs from database")
        return resume_texts, jd_texts, labels

def create_siamese_model(max_seq_length: int, embedding_dim: int = 100) -> Model:
    """
    Create a Siamese BiLSTM model for semantic text similarity.
    
    Args:
        max_seq_length (int): Maximum sequence length for input texts
        embedding_dim (int): Dimension of word embeddings
        
    Returns:
        Model: Compiled Siamese BiLSTM model
    """
    # Input layers
    input_1 = Input(shape=(max_seq_length, embedding_dim), name="input_1")
    input_2 = Input(shape=(max_seq_length, embedding_dim), name="input_2")
    
    # Shared BiLSTM layer
    lstm_layer = Bidirectional(LSTM(128, return_sequences=False))
    
    # Process both inputs through the same BiLSTM
    lstm_1 = lstm_layer(input_1)
    lstm_2 = lstm_layer(input_2)
    
    # Dropout for regularization
    dropout = Dropout(0.3)
    lstm_1 = dropout(lstm_1)
    lstm_2 = dropout(lstm_2)
    
    # Calculate cosine similarity between the two vectors
    def cosine_similarity(x):
        x1, x2 = x
        # Normalize the vectors
        x1_norm = tf.nn.l2_normalize(x1, axis=1)
        x2_norm = tf.nn.l2_normalize(x2, axis=1)
        # Calculate cosine similarity
        return tf.reduce_sum(x1_norm * x2_norm, axis=1, keepdims=True)
    
    similarity = Lambda(cosine_similarity)([lstm_1, lstm_2])
    
    # Output layer
    output = Dense(1, activation="sigmoid")(similarity)
    
    # Create model
    model = Model(inputs=[input_1, input_2], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def train_model(
    resume_texts: List[str],
    jd_texts: List[str],
    labels: List[bool],
    max_seq_length: int = 500,
    embedding_dim: int = 100,
    batch_size: int = 32,
    epochs: int = 10,
    validation_split: float = 0.2,
    model_dir: str = "models"
) -> Tuple[Model, Dict[str, Any]]:
    """
    Train the Siamese BiLSTM model on the provided data.
    
    Args:
        resume_texts (List[str]): List of resume texts
        jd_texts (List[str]): List of job description texts
        labels (List[bool]): List of binary labels indicating matches
        max_seq_length (int): Maximum sequence length for input texts
        embedding_dim (int): Dimension of word embeddings
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        validation_split (float): Proportion of data to use for validation
        model_dir (str): Directory to save the trained model
        
    Returns:
        Tuple[Model, Dict[str, Any]]: Trained model and training history
    """
    logger.info("Preparing data for training...")
    
    # Convert texts to sequences (placeholder - implement your tokenization)
    # For now, we'll use a simple character-level tokenization
    def text_to_sequence(text: str, max_length: int) -> np.ndarray:
        # Simple character-level tokenization
        chars = list(text[:max_length])
        # Pad or truncate
        if len(chars) < max_length:
            chars.extend([" "] * (max_length - len(chars)))
        # Convert to one-hot encoding (simplified)
        return np.array([[ord(c) % embedding_dim for c in chars]])
    
    # Convert texts to sequences
    X1 = np.array([text_to_sequence(text, max_seq_length) for text in resume_texts])
    X2 = np.array([text_to_sequence(text, max_seq_length) for text in jd_texts])
    y = np.array(labels)
    
    # Split data into train and validation sets
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1, X2, y, test_size=validation_split, random_state=42
    )
    
    logger.info(f"Training data shape: {X1_train.shape}")
    logger.info(f"Validation data shape: {X1_val.shape}")
    
    # Create and compile model
    model = create_siamese_model(max_seq_length, embedding_dim)
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "siamese_model.h5"),
            monitor="val_loss",
            save_best_only=True
        )
    ]
    
    # Train model
    logger.info("Training model...")
    history = model.fit(
        [X1_train, X2_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X1_val, X2_val], y_val),
        callbacks=callbacks
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = model.predict([X1_val, X2_val])
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred_binary),
        "precision": precision_score(y_val, y_pred_binary),
        "recall": recall_score(y_val, y_pred_binary),
        "f1": f1_score(y_val, y_pred_binary)
    }
    
    logger.info(f"Validation metrics: {metrics}")
    
    # Save model and metrics
    save_model(model, os.path.join(model_dir, "siamese_model_final.h5"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    
    return model, history.history

def predict_similarity(
    model: Model,
    resume_text: str,
    jd_text: str,
    max_seq_length: int = 500,
    embedding_dim: int = 100
) -> float:
    """
    Predict similarity score between a resume and a job description.
    
    Args:
        model (Model): Trained Siamese BiLSTM model
        resume_text (str): Resume text
        jd_text (str): Job description text
        max_seq_length (int): Maximum sequence length for input texts
        embedding_dim (int): Dimension of word embeddings
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Convert texts to sequences (same as in training)
    def text_to_sequence(text: str, max_length: int) -> np.ndarray:
        chars = list(text[:max_length])
        if len(chars) < max_length:
            chars.extend([" "] * (max_length - len(chars)))
        return np.array([[ord(c) % embedding_dim for c in chars]])
    
    X1 = text_to_sequence(resume_text, max_seq_length)
    X2 = text_to_sequence(jd_text, max_seq_length)
    
    # Make prediction
    similarity_score = model.predict([X1, X2])[0][0]
    return float(similarity_score)

def main():
    """Main function to train the Siamese BiLSTM model."""
    parser = argparse.ArgumentParser(description="Train Siamese BiLSTM model for resume-JD similarity")
    parser.add_argument("--db-path", type=str, default="data/preprocessed.db", help="Path to SQLite database")
    parser.add_argument("--max-seq-length", type=int, default=500, help="Maximum sequence length")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    
    args = parser.parse_args()
    
    # Load data
    resume_texts, jd_texts, labels = load_data(args.db_path)
    
    if not resume_texts or not jd_texts or not labels:
        logger.error("No labeled data found in the database")
        return
    
    # Train model
    model, history = train_model(
        resume_texts=resume_texts,
        jd_texts=jd_texts,
        labels=labels,
        max_seq_length=args.max_seq_length,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        model_dir=args.model_dir
    )
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 