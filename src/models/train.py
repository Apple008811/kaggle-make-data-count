"""
Model training for Make Data Count competition
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import wandb

from src.config.config import MODEL_CONFIG, TRAINING_CONFIG, get_data_path, get_experiment_path
from src.data.preprocess import load_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCitationDataset(Dataset):
    """Dataset for data citation classification"""
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        self.label_map = {'Primary': 0, 'Secondary': 1}
        self.num_labels = len(self.label_map)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert label to tensor
        label_tensor = torch.tensor(self.label_map.get(label, 0), dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }

class DataCitationClassifier(nn.Module):
    """BERT-based classifier for data citation classification"""
    
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return loss, logits

class ModelTrainer:
    """Trainer for data citation classification models"""
    
    def __init__(self, model_config: Dict, training_config: Dict):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize wandb if enabled
        if training_config.get('use_wandb', False):
            wandb.init(project="make-data-count", config=model_config)
    
    def prepare_data(self, data_path: Path) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data"""
        logger.info("Loading and preparing data...")
        
        # Load processed data
        df = load_data(data_path)
        
        # Filter rows with data citations
        df_with_citations = df[df['has_data_citation'] == True].copy()
        logger.info(f"Found {len(df_with_citations)} articles with data citations")
        
        # For now, we'll use a simple approach to create labels
        # In a real scenario, you'd have the actual labels from the training data
        df_with_citations['label'] = df_with_citations['primary_score'].apply(
            lambda x: 'Primary' if x > 0 else 'Secondary'
        )
        
        # Split data
        train_df, val_df = train_test_split(
            df_with_citations,
            test_size=self.training_config['validation_size'],
            random_state=self.training_config['random_state'],
            stratify=df_with_citations['label']
        )
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_config['bert_model_name'])
        
        # Create datasets
        train_dataset = DataCitationDataset(
            train_df['cleaned_text'].tolist(),
            train_df['label'].tolist(),
            tokenizer,
            self.model_config['max_length']
        )
        
        val_dataset = DataCitationDataset(
            val_df['cleaned_text'].tolist(),
            val_df['label'].tolist(),
            tokenizer,
            self.model_config['max_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader, tokenizer
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   tokenizer, experiment_name: str = "default"):
        """Train the model"""
        logger.info("Initializing model...")
        
        # Initialize model
        model = DataCitationClassifier(
            self.model_config['bert_model_name'],
            num_labels=2
        ).to(self.device)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        
        total_steps = len(train_loader) * self.model_config['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.model_config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training loop
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.model_config['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.model_config['num_epochs']}")
            
            # Training phase
            model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                loss, _ = model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    loss, logits = model(input_ids, attention_mask, labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1: {f1:.4f}")
            
            # Log to wandb
            if self.training_config.get('use_wandb', False):
                wandb.log({
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'f1_score': f1,
                    'epoch': epoch
                })
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                
                if self.training_config.get('save_best_model', True):
                    experiment_path = get_experiment_path(experiment_name)
                    experiment_path.mkdir(parents=True, exist_ok=True)
                    
                    model_path = experiment_path / "best_model.pt"
                    torch.save(model.state_dict(), model_path)
                    tokenizer.save_pretrained(experiment_path)
                    
                    logger.info(f"Saved best model with F1: {f1:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.training_config['early_stopping_patience']:
                logger.info("Early stopping triggered")
                break
        
        return model, tokenizer, best_f1

def main():
    """Main training function"""
    logger.info("Starting model training...")
    
    # Initialize trainer
    trainer = ModelTrainer(MODEL_CONFIG, TRAINING_CONFIG)
    
    # Prepare data
    train_path = get_data_path("train_processed.csv")
    if not train_path.exists():
        logger.error("Processed training data not found. Please run preprocessing first.")
        return
    
    train_loader, val_loader, tokenizer = trainer.prepare_data(train_path)
    
    # Train model
    experiment_name = f"bert_classifier_{MODEL_CONFIG['bert_model_name'].split('/')[-1]}"
    model, tokenizer, best_f1 = trainer.train_model(train_loader, val_loader, tokenizer, experiment_name)
    
    logger.info(f"Training completed! Best F1 Score: {best_f1:.4f}")
    
    # Save final model
    experiment_path = get_experiment_path(experiment_name)
    final_model_path = experiment_path / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    
    # Save training results
    results = {
        'best_f1_score': best_f1,
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG
    }
    
    with open(experiment_path / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Model and results saved to {experiment_path}")

if __name__ == "__main__":
    main() 