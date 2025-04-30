import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SiameseBiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 300,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Siamese BiLSTM with Attention model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_size (int): Size of word embeddings
            hidden_size (int): Size of LSTM hidden state
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            bidirectional (bool): Whether to use bidirectional LSTM
            device (str): Device to use for computation
        """
        super(SiameseBiLSTMAttention, self).__init__()
        
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def _init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for LSTM."""
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        return h0, c0
    
    def _attention_net(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to LSTM output.
        
        Args:
            lstm_output (torch.Tensor): Output from LSTM layer
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Context vector and attention weights
        """
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights
    
    def forward_one(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single input sequence.
        
        Args:
            x (torch.Tensor): Input sequence
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Context vector and attention weights
        """
        # Initialize hidden state
        batch_size = x.size(0)
        hidden = self._init_hidden(batch_size)
        
        # Embedding layer
        embedded = self.embedding(x)
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded, hidden)
        
        # Attention layer
        context, attention_weights = self._attention_net(lstm_out)
        
        return context, attention_weights
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for a pair of sequences.
        
        Args:
            x1 (torch.Tensor): First input sequence
            x2 (torch.Tensor): Second input sequence
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Similarity score and attention weights
        """
        # Get context vectors and attention weights for both sequences
        context1, attention_weights1 = self.forward_one(x1)
        context2, attention_weights2 = self.forward_one(x2)
        
        # Concatenate context vectors
        combined = torch.cat((context1, context2), dim=1)
        
        # Output layer
        similarity = self.fc(combined)
        
        # Return similarity score and attention weights
        attention_weights = {
            "sequence1": attention_weights1,
            "sequence2": attention_weights2
        }
        
        return similarity, attention_weights
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path (str): Path to save model
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            torch.save({
                'model_state_dict': self.state_dict(),
                'vocab_size': self.embedding.num_embeddings,
                'embedding_size': self.embedding.embedding_dim,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.lstm.dropout,
                'bidirectional': self.bidirectional,
                'device': self.device
            }, save_path)
            
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    @classmethod
    def load(cls, path: str) -> 'SiameseBiLSTMAttention':
        """
        Load model from disk.
        
        Args:
            path (str): Path to load model from
            
        Returns:
            SiameseBiLSTMAttention: Loaded model
        """
        try:
            checkpoint = torch.load(path)
            
            # Create model instance
            model = cls(
                vocab_size=checkpoint['vocab_size'],
                embedding_size=checkpoint['embedding_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout'],
                bidirectional=checkpoint['bidirectional'],
                device=checkpoint['device']
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def update_embeddings(self):
        """
        Update model embeddings with the latest processed JDs.
        This method should be called after new JDs are preprocessed.
        """
        try:
            # Get all processed JDs
            jd_dir = Path("data/processed")
            jd_embeddings = {}
            
            for jd_file in jd_dir.glob("*.json"):
                try:
                    with open(jd_file, 'r', encoding='utf-8') as f:
                        jd_data = json.load(f)
                        if "vector" in jd_data:
                            jd_embeddings[jd_data["id"]] = jd_data["vector"]
                except Exception as e:
                    print(f"Error loading JD embeddings from {jd_file}: {e}")
                    continue
            
            # Update model's JD embeddings
            self.jd_embeddings = jd_embeddings
            
            # Save updated embeddings
            embeddings_dir = Path("data/embeddings")
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            with open(embeddings_dir / "jd_embeddings.json", 'w') as f:
                json.dump(jd_embeddings, f)
            
            print(f"Updated embeddings for {len(jd_embeddings)} job descriptions")
            return True
            
        except Exception as e:
            print(f"Error updating embeddings: {e}")
            return False

class SimilarityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resume_texts: List[str],
        jd_texts: List[str],
        tokenizer,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize dataset for similarity training.
        
        Args:
            resume_texts (List[str]): List of resume texts
            jd_texts (List[str]): List of job description texts
            tokenizer: Tokenizer to use
            max_length (int): Maximum sequence length
            device (str): Device to use for computation
        """
        self.resume_texts = resume_texts
        self.jd_texts = jd_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        
        # Create pairs
        self.pairs = []
        for i, resume in enumerate(resume_texts):
            for j, jd in enumerate(jd_texts):
                self.pairs.append((i, j))
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of sequences and their similarity label.
        
        Args:
            idx (int): Index of the pair
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                First sequence, second sequence, and similarity label
        """
        resume_idx, jd_idx = self.pairs[idx]
        
        # Tokenize sequences
        resume_tokens = self.tokenizer(
            self.resume_texts[resume_idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        jd_tokens = self.tokenizer(
            self.jd_texts[jd_idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to tensors
        resume_tensor = resume_tokens['input_ids'].squeeze(0).to(self.device)
        jd_tensor = jd_tokens['input_ids'].squeeze(0).to(self.device)
        
        # For now, we'll use a simple binary label (1 for matching, 0 for non-matching)
        # In a real application, you might want to use more sophisticated matching criteria
        label = torch.tensor(1.0 if resume_idx == jd_idx else 0.0, device=self.device)
        
        return resume_tensor, jd_tensor, label

def train_model(
    model: SiameseBiLSTMAttention,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """
    Train the similarity model.
    
    Args:
        model (SiameseBiLSTMAttention): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader, optional): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (str): Device to use for computation
        
    Returns:
        Dict[str, List[float]]: Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(x1, x2)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for x1, x2, labels in val_loader:
                    outputs, _ = model(x1, x2)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    val_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted.squeeze() == labels).sum().item()
            
            val_loss /= len(val_loader)
            accuracy = correct / total
            
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(accuracy)
            
            logger.info(f'Epoch: {epoch+1}/{num_epochs}, '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val Accuracy: {accuracy:.4f}')
        else:
            logger.info(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
    
    return history

def main():
    """Main function to test the model."""
    # Example usage
    from transformers import AutoTokenizer
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Example data
    resume_texts = [
        "Experienced data scientist with expertise in machine learning and Python",
        "Senior software engineer with 5+ years of experience in web development"
    ]
    
    jd_texts = [
        "Looking for a data scientist with strong ML skills and Python experience",
        "Full-stack developer position requiring web development expertise"
    ]
    
    # Create dataset
    dataset = SimilarityDataset(resume_texts, jd_texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = SiameseBiLSTMAttention(
        vocab_size=tokenizer.vocab_size,
        embedding_size=300,
        hidden_size=256,
        num_layers=2,
        dropout=0.5
    )
    
    # Train model
    history = train_model(model, dataloader, num_epochs=5)
    
    # Save model
    model.save("models/similarity_model.pt")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 