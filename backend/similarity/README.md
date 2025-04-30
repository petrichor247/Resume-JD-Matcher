# Resume-JD Similarity Model

This module implements a Siamese BiLSTM with Attention model for semantic text similarity between resumes and job descriptions. The model is trained to identify how well a resume matches a job description based on semantic similarity.

## Features

- **Siamese BiLSTM with Attention**: Captures semantic relationships between resumes and job descriptions
- **MLflow Integration**: Tracks experiments, metrics, and model artifacts
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1 score, and ROC AUC metrics
- **Flexible Inference**: Find matching job descriptions for resumes or matching resumes for job descriptions
- **Attention Visualization**: See which parts of the texts are most important for similarity

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up MLflow (optional but recommended):

```bash
mlflow ui
```

## Usage

### Training

To train the model:

```bash
python train.py \
  --resume_dir /path/to/preprocessed/resumes \
  --jd_dir /path/to/preprocessed/jds \
  --output_dir models \
  --batch_size 32 \
  --num_epochs 10 \
  --learning_rate 0.001 \
  --embedding_size 300 \
  --hidden_size 256 \
  --num_layers 2 \
  --dropout 0.5 \
  --val_split 0.2 \
  --max_length 512 \
  --experiment_name resume-jd-similarity
```

### Evaluation

To evaluate the model:

```bash
python evaluate.py \
  --model_path /path/to/trained/model.pt \
  --resume_dir /path/to/preprocessed/resumes \
  --jd_dir /path/to/preprocessed/jds \
  --batch_size 32 \
  --max_length 512 \
  --experiment_name resume-jd-similarity-evaluation
```

### Inference

To run inference with the trained model:

```bash
python inference.py \
  --model_path /path/to/trained/model.pt \
  --resume_dir /path/to/preprocessed/resumes \
  --jd_dir /path/to/preprocessed/jds \
  --output_dir results \
  --top_k 5 \
  --threshold 0.5
```

## Model Architecture

The model uses a Siamese BiLSTM with Attention architecture:

1. **Embedding Layer**: Converts tokenized text to dense vectors
2. **BiLSTM Layer**: Processes text in both directions to capture context
3. **Attention Layer**: Identifies important parts of the text
4. **Similarity Layer**: Computes similarity between resume and job description

## MLflow Integration

The model training and evaluation are integrated with MLflow for experiment tracking:

- **Parameters**: Model architecture, training hyperparameters
- **Metrics**: Training loss, validation loss, accuracy, precision, recall, F1 score, ROC AUC
- **Artifacts**: Trained model, confusion matrix, ROC curve, training history

To view the MLflow UI:

```bash
mlflow ui
```

Then open your browser at http://localhost:5000

## Example

```python
from inference import SimilarityPredictor

# Initialize predictor
predictor = SimilarityPredictor(
    model_path="models/similarity_model.pt",
    threshold=0.5
)

# Predict similarity
resume_text = "Experienced data scientist with expertise in machine learning and Python"
jd_text = "Looking for a data scientist with strong ML skills and Python experience"

similarity_score, attention_weights = predictor.predict_similarity(resume_text, jd_text)
print(f"Similarity Score: {similarity_score:.4f}")

# Find matching JDs for a resume
matches = predictor.find_matching_jds(
    resume_text=resume_text,
    jd_texts=[jd_text, "Senior software engineer position"],
    top_k=2
)

for match in matches:
    print(f"JD: {match['jd_text']}")
    print(f"Similarity Score: {match['similarity_score']:.4f}")
    print(f"Is Match: {match['is_match']}")
    print("---")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 