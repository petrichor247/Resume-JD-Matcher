import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import os
import mlflow
import mlflow.tensorflow
from datetime import datetime
import json

class DistanceLayer(layers.Layer):
    """Custom layer for calculating the distance between two embeddings."""
    
    def __init__(self, **kwargs):
        super(DistanceLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        """Calculate the absolute difference between two embeddings."""
        embedding_a, embedding_b = inputs
        return tf.keras.backend.abs(embedding_a - embedding_b)
    
    def get_config(self):
        """Get the layer configuration."""
        config = super(DistanceLayer, self).get_config()
        return config

class SiameseResumeMatcher:
    def __init__(self, embedding_dim=128, input_shape=(5000,)):
        """Initialize the Siamese network for resume-job matching."""
        self.embedding_dim = embedding_dim
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the Siamese network architecture."""
        # Input layers
        input_a = Input(shape=self.input_shape, name='input_a')
        input_b = Input(shape=self.input_shape, name='input_b')
        
        # Base network (shared weights)
        base_network = self._create_base_network()
        
        # Process inputs through the base network
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # Calculate distance between the two embeddings using custom layer
        distance = DistanceLayer(name='distance')([processed_a, processed_b])
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(distance)
        
        # Create model
        model = Model(inputs=[input_a, input_b], outputs=output)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_base_network(self):
        """Create the base network with shared weights."""
        input_layer = Input(shape=self.input_shape)
        
        x = layers.Dense(512, activation='relu')(input_layer)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.embedding_dim, activation='relu')(x)
        
        return Model(inputs=input_layer, outputs=x)
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10, experiment_name="resume_matcher"):
        """Train the model with MLflow tracking."""
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("embedding_dim", self.embedding_dim)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            
            # Train the model
            history = self.model.fit(
                [X_train[:, 0], X_train[:, 1]],
                y_train,
                validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1
            )
            
            # Log metrics
            for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("train_accuracy", acc, step=epoch)
            
            for epoch, (val_loss, val_acc) in enumerate(zip(history.history['val_loss'], history.history['val_accuracy'])):
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            # Log the model
            mlflow.tensorflow.log_model(self.model, "model")
            
            return history
    
    def predict_similarity(self, resume_embedding, job_embedding):
        """Predict similarity between a resume and a job."""
        # Reshape inputs for the model
        resume_embedding = resume_embedding.reshape(1, -1)
        job_embedding = job_embedding.reshape(1, -1)
        
        # Create input pairs
        input_a = np.vstack([resume_embedding])
        input_b = np.vstack([job_embedding])
        
        # Predict similarity
        similarity = self.model.predict([input_a, input_b])[0][0]
        
        return float(similarity)
    
    def save_model(self, path):
        """Save the model to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Ensure the path has .keras extension
        if not path.endswith('.keras'):
            path = f"{path}.keras"
        self.model.save(path)
        
        # Save model metadata
        metadata = {
            "embedding_dim": self.embedding_dim,
            "input_shape": self.input_shape,
            "model_type": "siamese"
        }
        metadata_path = f"{path}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load_model(cls, path, embedding_dim=None, input_shape=None):
        """Load a saved model from disk."""
        # Ensure the path has .keras extension
        if not path.endswith('.keras'):
            path = f"{path}.keras"
            
        # Load metadata if available
        metadata_path = f"{path}.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                embedding_dim = metadata.get("embedding_dim", embedding_dim)
                input_shape = metadata.get("input_shape", input_shape)
        
        # Create model instance
        model = cls(embedding_dim=embedding_dim, input_shape=input_shape)
        
        # Register custom objects
        custom_objects = {"DistanceLayer": DistanceLayer}
        
        # Load the saved model
        model.model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        return model 