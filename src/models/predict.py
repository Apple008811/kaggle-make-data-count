"""
Prediction module for Make Data Count competition
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from src.config.config import MODEL_CONFIG, get_data_path, get_experiment_path
from src.data.preprocess import load_data
from src.models.train import DataCitationClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCitationPredictor:
    """Predictor for data citation classification"""
    
    def __init__(self, model_path: Path, tokenizer_path: Path, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = self._load_model()
        
        # Label mapping
        self.label_map = {0: 'Primary', 1: 'Secondary'}
        
    def _load_model(self) -> DataCitationClassifier:
        """Load the trained model"""
        model = DataCitationClassifier(
            MODEL_CONFIG['bert_model_name'],
            num_labels=2
        ).to(self.device)
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        
        return model
    
    def predict_single(self, text: str) -> Tuple[str, float]:
        """Predict class and confidence for a single text"""
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MODEL_CONFIG['max_length'],
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
        
        return self.label_map[predicted_class], confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict classes and confidences for a batch of texts"""
        predictions = []
        
        for text in texts:
            pred_class, confidence = self.predict_single(text)
            predictions.append((pred_class, confidence))
        
        return predictions

def generate_predictions(test_data_path: Path, model_experiment: str = "default") -> pd.DataFrame:
    """Generate predictions for test data"""
    logger.info("Loading test data...")
    
    # Load test data
    test_df = load_data(test_data_path)
    
    # Load model
    experiment_path = get_experiment_path(model_experiment)
    model_path = experiment_path / "best_model.pt"
    tokenizer_path = experiment_path
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return None
    
    predictor = DataCitationPredictor(model_path, tokenizer_path)
    
    # Filter articles with potential data citations
    test_df_with_citations = test_df[test_df['has_data_citation'] == True].copy()
    logger.info(f"Found {len(test_df_with_citations)} test articles with potential data citations")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []
    
    for idx, row in test_df_with_citations.iterrows():
        text = row['cleaned_text']
        article_id = row['article_id']
        
        # Get DOIs from the text
        dois = row.get('dois', [])
        
        if not dois:
            # If no DOIs found, skip this article
            continue
        
        # Predict for each DOI
        for doi in dois:
            pred_class, confidence = predictor.predict_single(text)
            
            predictions.append({
                'article_id': article_id,
                'dataset_id': doi,
                'type': pred_class,
                'confidence': confidence
            })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Remove duplicates (same article_id, dataset_id, type)
    predictions_df = predictions_df.drop_duplicates(
        subset=['article_id', 'dataset_id', 'type']
    )
    
    # Add row_id
    predictions_df['row_id'] = range(len(predictions_df))
    
    # Reorder columns to match submission format
    predictions_df = predictions_df[['row_id', 'article_id', 'dataset_id', 'type']]
    
    logger.info(f"Generated {len(predictions_df)} predictions")
    
    return predictions_df

def create_submission(predictions_df: pd.DataFrame, output_path: Path) -> None:
    """Create submission file"""
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

def main():
    """Main prediction function"""
    logger.info("Starting prediction...")
    
    # Check if test data exists
    test_path = get_data_path("test_processed.csv")
    if not test_path.exists():
        logger.error("Processed test data not found. Please run preprocessing first.")
        return
    
    # Generate predictions
    predictions_df = generate_predictions(test_path)
    
    if predictions_df is None:
        logger.error("Failed to generate predictions")
        return
    
    # Create submission
    submission_path = get_data_path("submission.csv")
    create_submission(predictions_df, submission_path)
    
    # Print summary
    logger.info("Prediction Summary:")
    logger.info(f"Total predictions: {len(predictions_df)}")
    logger.info(f"Primary predictions: {len(predictions_df[predictions_df['type'] == 'Primary'])}")
    logger.info(f"Secondary predictions: {len(predictions_df[predictions_df['type'] == 'Secondary'])}")
    
    # Show sample predictions
    logger.info("\nSample predictions:")
    print(predictions_df.head(10))

if __name__ == "__main__":
    main() 