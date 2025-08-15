"""
Configuration file for Make Data Count competition
"""
import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

# Model configuration
MODEL_CONFIG = {
    "bert_model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2,
}

# Training configuration
TRAINING_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.1,
    "early_stopping_patience": 3,
    "save_best_model": True,
    "use_wandb": False,  # Set to True if using Weights & Biases
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "use_doi_patterns": True,
    "use_keyword_patterns": True,
    "use_section_analysis": True,
    "use_citation_patterns": True,
    "min_doi_confidence": 0.8,
}

# Keywords for data citation detection
DATA_CITATION_KEYWORDS = [
    "dataset", "data", "database", "repository", "archive",
    "zenodo", "figshare", "dryad", "dataverse", "github",
    "supplementary", "supplemental", "available", "obtained",
    "downloaded", "retrieved", "accessed", "collected",
    "generated", "created", "produced", "derived"
]

# DOI patterns
DOI_PATTERNS = [
    r"10\.\d{4,}/[-._;()/:\w]+",
    r"https?://doi\.org/10\.\d{4,}/[-._;()/:\w]+",
    r"https?://zenodo\.org/record/\d+",
    r"https?://figshare\.com/articles/\d+",
    r"https?://datadryad\.org/stash/dataset/doi:10\.\d{4,}/[-._;()/:\w]+",
]

# Primary data indicators
PRIMARY_INDICATORS = [
    "generated", "created", "produced", "collected", "measured",
    "experimental", "laboratory", "field", "survey", "sampling",
    "our data", "this study", "we collected", "we generated",
    "primary data", "raw data", "experimental data"
]

# Secondary data indicators
SECONDARY_INDICATORS = [
    "obtained from", "downloaded from", "retrieved from",
    "available at", "accessed from", "derived from",
    "reused", "existing", "published", "publicly available",
    "secondary data", "external data", "previous study",
    "literature", "database", "repository"
]

# Submission configuration
SUBMISSION_CONFIG = {
    "output_file": "submission.csv",
    "include_confidence": False,
    "min_confidence_threshold": 0.5,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "competition.log"
}

def get_data_path(filename: str) -> Path:
    """Get the full path to a data file"""
    return DATA_DIR / filename

def get_experiment_path(experiment_name: str) -> Path:
    """Get the path for experiment outputs"""
    return EXPERIMENTS_DIR / experiment_name

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [DATA_DIR, EXPERIMENTS_DIR, NOTEBOOKS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True) 