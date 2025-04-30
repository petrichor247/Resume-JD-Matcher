# model_trainer.py

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import mlflow
import mlflow.tensorflow

from models.siamese_model import SiameseResumeMatcher
from data_processor import DataProcessor

# Optional: optimize CPU thread usage
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)


class ModelTrainer:
    def __init__(self, data_dir="data", model_dir="models", mlflow_dir="mlflow_logs"):
        """Initialize the model trainer."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.mlflow_dir = mlflow_dir

        # Set up MLflow
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow_path = Path(mlflow_dir).absolute()
        mlflow.set_tracking_uri(str(mlflow_path))

        # Initialize data processor
        self.data_processor = DataProcessor(data_dir=data_dir)

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

    def train_model(self, embedding_dim=128, batch_size=32, epochs=10, num_samples=1000):
        """Train the Siamese network model."""
        try:
            print("Starting model training...")
            
            # Prepare training data
            X_train, y_train, X_val, y_val = self.data_processor.prepare_training_data(num_samples=num_samples)

            if X_train is None or len(X_train) == 0:
                print("Not enough data to train the model.")
                return None

            input_shape = (X_train.shape[2],)
            print(f"Input shape: {input_shape}")

            # Initialize model
            model = SiameseResumeMatcher(embedding_dim=embedding_dim, input_shape=input_shape)

            # Configure MLflow autologging
            mlflow.tensorflow.autolog(log_models=False)

            # Train model
            history = model.train(
                X_train, y_train, X_val, y_val,
                batch_size=batch_size,
                epochs=epochs,
                experiment_name="resume_matcher"
            )

            # Save model
            model_path = os.path.join(self.model_dir, "siamese_model")
            model.save_model(model_path)

            # Update training time
            self.data_processor.update_last_training_time()

            # Save metadata
            metadata = {
                "embedding_dim": embedding_dim,
                "input_shape": input_shape,
                "batch_size": batch_size,
                "epochs": epochs,
                "num_samples": num_samples,
                "model_path": model_path,
                "training_date": datetime.now().isoformat()
            }

            metadata_path = os.path.join(self.model_dir, "model_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print("Model training completed successfully!")
            return model

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise

    def load_model(self):
        """Load the trained model."""
        try:
            model_path = os.path.join(self.model_dir, "siamese_model.keras")
            metadata_path = os.path.join(self.model_dir, "model_metadata.json")

            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                print("Model or metadata file not found.")
                return None

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            model = SiameseResumeMatcher.load_model(
                model_path,
                embedding_dim=metadata.get("embedding_dim", 128),
                input_shape=tuple(metadata.get("input_shape", (5000,)))
            )

            print("Model loaded successfully!")
            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def find_matches(self, resume_id, top_k=5):
        """Find matching jobs for a given resume."""
        try:
            model = self.load_model()
            if model is None:
                print("No trained model found, using fallback matching.")
                return self._fallback_matching(resume_id, top_k)

            resume_embedding = self.data_processor.get_resume_embedding(resume_id)
            if resume_embedding is None:
                raise ValueError("Resume embedding not found")

            job_embeddings = {}
            recent_jobs = self.data_processor.get_recent_jobs()

            for job in recent_jobs:
                job_id = job['id']
                embedding = self.data_processor.get_job_embedding(job_id)
                if embedding is not None:
                    job_embeddings[job_id] = embedding

            scores = []
            for job_id, job_embedding in job_embeddings.items():
                try:
                    similarity = model.predict_similarity(resume_embedding, job_embedding)
                    if not np.isnan(similarity):
                        job_details = self.data_processor.get_job_details(job_id)
                        recency_score = 0.5
                        if job_details and 'posting_date' in job_details:
                            posting_date = datetime.fromisoformat(job_details['posting_date'])
                            days_old = (datetime.now() - posting_date).days
                            recency_score = 1.0 / (1.0 + days_old / 30.0)

                        combined_score = 0.7 * similarity + 0.3 * recency_score
                        scores.append((job_id, combined_score))
                except Exception:
                    continue

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        except Exception as e:
            print(f"Error in find_matches: {str(e)}")
            return self._fallback_matching(resume_id, top_k)

    def _fallback_matching(self, resume_id, top_k=5):
        """Fallback matching using cosine similarity."""
        try:
            resume_embedding = self.data_processor.get_resume_embedding(resume_id)
            if resume_embedding is None:
                raise ValueError("Resume embedding not found")

            scores = []
            for job in self.data_processor.get_recent_jobs():
                job_id = job['id']
                job_embedding = self.data_processor.get_job_embedding(job_id)
                if job_embedding is not None:
                    try:
                        similarity = np.inner(resume_embedding.flatten(), job_embedding.flatten())
                        similarity /= (np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding))
                        if not np.isnan(similarity):
                            scores.append((job_id, float(similarity)))
                    except Exception:
                        continue

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        except Exception as e:
            print(f"Error in fallback matching: {str(e)}")
            return []

    def _log_to_dvc(self):
        """Placeholder for DVC logging."""
        pass
