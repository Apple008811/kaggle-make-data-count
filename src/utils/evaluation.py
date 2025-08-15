"""
Evaluation utilities for Make Data Count competition
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for data citation classification models"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true: List, y_pred: List, y_prob: Optional[List] = None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        metrics['precision_primary'] = precision_score(y_true, y_pred, pos_label='Primary', zero_division=0)
        metrics['precision_secondary'] = precision_score(y_true, y_pred, pos_label='Secondary', zero_division=0)
        metrics['recall_primary'] = recall_score(y_true, y_pred, pos_label='Primary', zero_division=0)
        metrics['recall_secondary'] = recall_score(y_true, y_pred, pos_label='Secondary', zero_division=0)
        metrics['f1_primary'] = f1_score(y_true, y_pred, pos_label='Primary', zero_division=0)
        metrics['f1_secondary'] = f1_score(y_true, y_pred, pos_label='Secondary', zero_division=0)
        
        return metrics
    
    def generate_classification_report(self, y_true: List, y_pred: List) -> str:
        """Generate detailed classification report"""
        return classification_report(y_true, y_pred, target_names=['Primary', 'Secondary'])
    
    def create_confusion_matrix(self, y_true: List, y_pred: List, save_path: Optional[Path] = None) -> np.ndarray:
        """Create and optionally save confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred, labels=['Primary', 'Secondary'])
        
        if save_path:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Primary', 'Secondary'],
                       yticklabels=['Primary', 'Secondary'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return cm
    
    def evaluate_predictions(self, predictions_df: pd.DataFrame, 
                           ground_truth_df: pd.DataFrame) -> Dict:
        """Evaluate predictions against ground truth"""
        logger.info("Evaluating predictions...")
        
        # Merge predictions with ground truth
        merged_df = predictions_df.merge(
            ground_truth_df, 
            on=['article_id', 'dataset_id'], 
            how='inner',
            suffixes=('_pred', '_true')
        )
        
        if len(merged_df) == 0:
            logger.warning("No matching predictions found for evaluation")
            return {}
        
        # Extract true and predicted labels
        y_true = merged_df['type_true'].tolist()
        y_pred = merged_df['type_pred'].tolist()
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Generate detailed report
        report = self.generate_classification_report(y_true, y_pred)
        
        # Store results
        self.metrics = metrics
        
        logger.info(f"Evaluation completed. F1 Score: {metrics['f1_weighted']:.4f}")
        logger.info(f"Precision: {metrics['precision_weighted']:.4f}")
        logger.info(f"Recall: {metrics['recall_weighted']:.4f}")
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'evaluated_samples': len(merged_df)
        }
    
    def save_evaluation_results(self, results: Dict, output_path: Path) -> None:
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {output_path}")

def evaluate_submission(submission_path: Path, ground_truth_path: Path, 
                       output_dir: Path) -> Dict:
    """Evaluate a submission file against ground truth"""
    logger.info("Evaluating submission...")
    
    # Load data
    submission_df = pd.read_csv(submission_path)
    ground_truth_df = pd.read_csv(ground_truth_path)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate predictions
    results = evaluator.evaluate_predictions(submission_df, ground_truth_df)
    
    if not results:
        return {}
    
    # Create confusion matrix
    merged_df = submission_df.merge(
        ground_truth_df, 
        on=['article_id', 'dataset_id'], 
        how='inner',
        suffixes=('_pred', '_true')
    )
    
    y_true = merged_df['type_true'].tolist()
    y_pred = merged_df['type_pred'].tolist()
    
    cm_path = output_dir / "confusion_matrix.png"
    evaluator.create_confusion_matrix(y_true, y_pred, cm_path)
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    evaluator.save_evaluation_results(results, results_path)
    
    return results

def analyze_predictions(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze prediction distribution and characteristics"""
    logger.info("Analyzing predictions...")
    
    # Basic statistics
    total_predictions = len(predictions_df)
    primary_count = len(predictions_df[predictions_df['type'] == 'Primary'])
    secondary_count = len(predictions_df[predictions_df['type'] == 'Secondary'])
    
    # Create analysis report
    analysis = {
        'total_predictions': total_predictions,
        'primary_predictions': primary_count,
        'secondary_predictions': secondary_count,
        'primary_percentage': (primary_count / total_predictions) * 100 if total_predictions > 0 else 0,
        'secondary_percentage': (secondary_count / total_predictions) * 100 if total_predictions > 0 else 0,
        'unique_articles': predictions_df['article_id'].nunique(),
        'unique_datasets': predictions_df['dataset_id'].nunique(),
    }
    
    # Save analysis
    analysis_path = output_dir / "prediction_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Prediction distribution
    plt.subplot(2, 2, 1)
    predictions_df['type'].value_counts().plot(kind='bar')
    plt.title('Prediction Distribution')
    plt.ylabel('Count')
    
    # Articles per prediction type
    plt.subplot(2, 2, 2)
    article_counts = predictions_df.groupby('type')['article_id'].nunique()
    article_counts.plot(kind='bar')
    plt.title('Unique Articles per Prediction Type')
    plt.ylabel('Number of Articles')
    
    # Datasets per prediction type
    plt.subplot(2, 2, 3)
    dataset_counts = predictions_df.groupby('type')['dataset_id'].nunique()
    dataset_counts.plot(kind='bar')
    plt.title('Unique Datasets per Prediction Type')
    plt.ylabel('Number of Datasets')
    
    # Predictions per article
    plt.subplot(2, 2, 4)
    predictions_per_article = predictions_df['article_id'].value_counts()
    plt.hist(predictions_per_article, bins=20, alpha=0.7)
    plt.title('Predictions per Article')
    plt.xlabel('Number of Predictions')
    plt.ylabel('Number of Articles')
    
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction analysis saved to {output_dir}")

def main():
    """Main evaluation function"""
    from src.config.config import get_data_path, get_experiment_path
    
    # Example usage
    submission_path = get_data_path("submission.csv")
    ground_truth_path = get_data_path("ground_truth.csv")  # This would be the actual test labels
    
    if not submission_path.exists():
        logger.error("Submission file not found")
        return
    
    if not ground_truth_path.exists():
        logger.warning("Ground truth file not found. Running prediction analysis only.")
        predictions_df = pd.read_csv(submission_path)
        output_dir = get_experiment_path("evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        analyze_predictions(predictions_df, output_dir)
        return
    
    # Run full evaluation
    output_dir = get_experiment_path("evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = evaluate_submission(submission_path, ground_truth_path, output_dir)
    
    if results:
        predictions_df = pd.read_csv(submission_path)
        analyze_predictions(predictions_df, output_dir)
        
        logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 