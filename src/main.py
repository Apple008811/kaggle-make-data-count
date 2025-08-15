"""
Main pipeline for Make Data Count competition
"""
import argparse
import logging
from pathlib import Path
import sys

from src.config.config import create_directories, get_data_path
from src.data.preprocess import preprocess_train_data, preprocess_test_data, create_sample_data
from src.features.feature_engineering import engineer_features_for_dataset, select_features
from src.models.train import main as train_model
from src.models.predict import main as predict_model
from src.utils.evaluation import main as evaluate_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_preprocessing():
    """Run data preprocessing"""
    logger.info("=== Starting Data Preprocessing ===")
    
    # Create sample data if no real data exists
    train_path = get_data_path("train.csv")
    if not train_path.exists():
        logger.info("No training data found. Creating sample data...")
        create_sample_data()
    
    # Preprocess training data
    try:
        train_df = preprocess_train_data()
        logger.info("Training data preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Error preprocessing training data: {e}")
        return False
    
    # Preprocess test data if it exists
    test_path = get_data_path("test.csv")
    if test_path.exists():
        try:
            test_df = preprocess_test_data()
            logger.info("Test data preprocessing completed successfully!")
        except Exception as e:
            logger.error(f"Error preprocessing test data: {e}")
            return False
    
    return True

def run_feature_engineering():
    """Run feature engineering"""
    logger.info("=== Starting Feature Engineering ===")
    
    try:
        from src.data.preprocess import load_data
        
        # Load processed data
        train_path = get_data_path("train_processed.csv")
        if not train_path.exists():
            logger.error("Processed training data not found. Please run preprocessing first.")
            return False
        
        df = load_data(train_path)
        
        # Engineer features
        features_df = engineer_features_for_dataset(df)
        
        # Select features
        features_df = select_features(features_df, method='correlation')
        
        # Combine with original data
        result_df = df.join(features_df)
        
        # Save enhanced dataset
        output_path = get_data_path("train_with_features.csv")
        result_df.to_csv(output_path, index=False)
        
        logger.info(f"Feature engineering completed. Enhanced dataset saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return False

def run_training():
    """Run model training"""
    logger.info("=== Starting Model Training ===")
    
    try:
        train_model()
        logger.info("Model training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return False

def run_prediction():
    """Run model prediction"""
    logger.info("=== Starting Model Prediction ===")
    
    try:
        predict_model()
        logger.info("Model prediction completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in model prediction: {e}")
        return False

def run_evaluation():
    """Run model evaluation"""
    logger.info("=== Starting Model Evaluation ===")
    
    try:
        evaluate_model()
        logger.info("Model evaluation completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        return False

def run_full_pipeline():
    """Run the complete pipeline"""
    logger.info("=== Starting Full Pipeline ===")
    
    # Create necessary directories
    create_directories()
    
    # Step 1: Preprocessing
    if not run_preprocessing():
        logger.error("Pipeline failed at preprocessing step")
        return False
    
    # Step 2: Feature Engineering
    if not run_feature_engineering():
        logger.error("Pipeline failed at feature engineering step")
        return False
    
    # Step 3: Model Training
    if not run_training():
        logger.error("Pipeline failed at training step")
        return False
    
    # Step 4: Model Prediction
    if not run_prediction():
        logger.error("Pipeline failed at prediction step")
        return False
    
    # Step 5: Model Evaluation
    if not run_evaluation():
        logger.error("Pipeline failed at evaluation step")
        return False
    
    logger.info("=== Full Pipeline Completed Successfully! ===")
    return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Make Data Count Competition Pipeline')
    parser.add_argument('--step', choices=['preprocess', 'features', 'train', 'predict', 'evaluate', 'full'],
                       default='full', help='Pipeline step to run')
    parser.add_argument('--data-dir', type=str, help='Directory containing data files')
    parser.add_argument('--experiment-name', type=str, default='default',
                       help='Name for the experiment')
    
    args = parser.parse_args()
    
    # Set up data directory if specified
    if args.data_dir:
        from src.config.config import DATA_DIR
        DATA_DIR = Path(args.data_dir)
    
    logger.info(f"Running pipeline step: {args.step}")
    
    # Run the specified step
    if args.step == 'preprocess':
        success = run_preprocessing()
    elif args.step == 'features':
        success = run_feature_engineering()
    elif args.step == 'train':
        success = run_training()
    elif args.step == 'predict':
        success = run_prediction()
    elif args.step == 'evaluate':
        success = run_evaluation()
    elif args.step == 'full':
        success = run_full_pipeline()
    else:
        logger.error(f"Unknown step: {args.step}")
        return False
    
    if success:
        logger.info(f"Step '{args.step}' completed successfully!")
    else:
        logger.error(f"Step '{args.step}' failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 