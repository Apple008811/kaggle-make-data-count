"""
Feature engineering for Make Data Count competition
"""
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
from collections import Counter
import spacy
from textblob import TextBlob

from src.config.config import (
    DATA_CITATION_KEYWORDS, PRIMARY_INDICATORS, SECONDARY_INDICATORS,
    FEATURE_CONFIG
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for scientific text data"""
    
    def __init__(self):
        # Load spaCy model for NLP features
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from text"""
        if not text or pd.isna(text):
            return {}
        
        doc = self.nlp(text)
        
        # Basic linguistic features
        features = {
            'sentence_count': len(list(doc.sents)),
            'token_count': len(doc),
            'avg_sentence_length': len(doc) / max(len(list(doc.sents)), 1),
            'noun_count': len([token for token in doc if token.pos_ == 'NOUN']),
            'verb_count': len([token for token in doc if token.pos_ == 'VERB']),
            'adj_count': len([token for token in doc if token.pos_ == 'ADJ']),
            'entity_count': len(doc.ents),
        }
        
        # Named entity features
        entity_types = Counter([ent.label_ for ent in doc.ents])
        for entity_type in ['PERSON', 'ORG', 'GPE', 'DATE', 'CARDINAL']:
            features[f'{entity_type.lower()}_count'] = entity_types.get(entity_type, 0)
        
        return features
    
    def extract_scientific_features(self, text: str) -> Dict[str, Any]:
        """Extract scientific domain-specific features"""
        text_lower = text.lower()
        
        # Scientific terminology
        scientific_terms = [
            'methodology', 'experiment', 'analysis', 'results', 'conclusion',
            'hypothesis', 'theory', 'model', 'framework', 'algorithm',
            'statistical', 'significant', 'correlation', 'regression',
            'sample', 'population', 'variable', 'parameter', 'coefficient'
        ]
        
        features = {}
        for term in scientific_terms:
            features[f'scientific_{term}_count'] = text_lower.count(term)
        
        # Citation patterns
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[[^\]]*\d{4}[^\]]*\]',  # [Author, 2023]
            r'et al\.',  # et al.
            r'doi:',  # DOI references
            r'https?://',  # URLs
        ]
        
        for i, pattern in enumerate(citation_patterns):
            features[f'citation_pattern_{i}_count'] = len(re.findall(pattern, text, re.IGNORECASE))
        
        return features
    
    def extract_context_features(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract context-based features"""
        features = {}
        
        # Section-based features
        for section_name, section_text in sections.items():
            if section_text:
                features[f'{section_name}_length'] = len(section_text)
                features[f'{section_name}_word_count'] = len(section_text.split())
                
                # Data citation indicators in each section
                primary_count = sum(section_text.lower().count(indicator) 
                                  for indicator in PRIMARY_INDICATORS)
                secondary_count = sum(section_text.lower().count(indicator) 
                                    for indicator in SECONDARY_INDICATORS)
                
                features[f'{section_name}_primary_indicators'] = primary_count
                features[f'{section_name}_secondary_indicators'] = secondary_count
        
        # Context window features
        if 'methods' in sections and sections['methods']:
            features['methods_has_data'] = any(
                keyword in sections['methods'].lower() 
                for keyword in DATA_CITATION_KEYWORDS
            )
        
        if 'results' in sections and sections['results']:
            features['results_has_data'] = any(
                keyword in sections['results'].lower() 
                for keyword in DATA_CITATION_KEYWORDS
            )
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract sentiment and tone features"""
        if not text or pd.isna(text):
            return {}
        
        blob = TextBlob(text)
        
        features = {
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
        }
        
        # Sentiment by sentence
        sentences = blob.sentences
        if sentences:
            polarities = [s.sentiment.polarity for s in sentences]
            subjectivities = [s.sentiment.subjectivity for s in sentences]
            
            features.update({
                'avg_sentence_polarity': np.mean(polarities),
                'std_sentence_polarity': np.std(polarities),
                'avg_sentence_subjectivity': np.mean(subjectivities),
                'std_sentence_subjectivity': np.std(subjectivities),
            })
        
        return features
    
    def extract_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract structural features from text"""
        features = {}
        
        # Paragraph structure
        paragraphs = text.split('\n\n')
        features['paragraph_count'] = len(paragraphs)
        features['avg_paragraph_length'] = np.mean([len(p) for p in paragraphs]) if paragraphs else 0
        
        # List and enumeration features
        features['list_count'] = len(re.findall(r'^\s*[\-\*•]\s', text, re.MULTILINE))
        features['numbered_list_count'] = len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))
        
        # Table and figure references
        features['table_ref_count'] = len(re.findall(r'table\s+\d+', text, re.IGNORECASE))
        features['figure_ref_count'] = len(re.findall(r'figure\s+\d+', text, re.IGNORECASE))
        features['equation_ref_count'] = len(re.findall(r'equation\s+\d+', text, re.IGNORECASE))
        
        return features
    
    def extract_all_features(self, text: str, sections: Dict[str, str] = None) -> Dict[str, Any]:
        """Extract all features from text"""
        if sections is None:
            sections = {}
        
        features = {}
        
        # Extract different types of features
        features.update(self.extract_linguistic_features(text))
        features.update(self.extract_scientific_features(text))
        features.update(self.extract_context_features(text, sections))
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_structural_features(text))
        
        return features

def engineer_features_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for entire dataset"""
    logger.info("Starting feature engineering...")
    
    feature_engineer = FeatureEngineer()
    
    # Extract features for each row
    all_features = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing row {idx}/{len(df)}")
        
        text = row.get('cleaned_text', '')
        sections = {
            'abstract': row.get('abstract', ''),
            'methods': row.get('methods', ''),
            'results': row.get('results', ''),
            'discussion': row.get('discussion', ''),
            'conclusion': row.get('conclusion', '')
        }
        
        features = feature_engineer.extract_all_features(text, sections)
        all_features.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Fill NaN values
    features_df = features_df.fillna(0)
    
    logger.info(f"Feature engineering completed. Generated {len(features_df.columns)} features")
    
    return features_df

def select_features(features_df: pd.DataFrame, method: str = 'correlation') -> pd.DataFrame:
    """Select most important features"""
    logger.info(f"Selecting features using {method} method...")
    
    if method == 'correlation':
        # Remove highly correlated features
        corr_matrix = features_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.95
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        features_df = features_df.drop(columns=to_drop)
        logger.info(f"Dropped {len(to_drop)} highly correlated features")
    
    elif method == 'variance':
        # Remove low variance features
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        features_df = pd.DataFrame(
            selector.fit_transform(features_df),
            columns=features_df.columns[selector.get_support()]
        )
        logger.info(f"Selected {len(features_df.columns)} features based on variance")
    
    return features_df

def main():
    """Main feature engineering function"""
    logger.info("Starting feature engineering pipeline...")
    
    # Load processed data
    from src.data.preprocess import load_data
    from src.config.config import get_data_path
    
    train_path = get_data_path("train_processed.csv")
    if not train_path.exists():
        logger.error("Processed training data not found. Please run preprocessing first.")
        return
    
    df = load_data(train_path)
    
    # Engineer features
    features_df = engineer_features_for_dataset(df)
    
    # Select features
    features_df = select_features(features_df, method='correlation')
    
    # Combine with original data
    result_df = pd.concat([df, features_df], axis=1)
    
    # Save enhanced dataset
    output_path = get_data_path("train_with_features.csv")
    result_df.to_csv(output_path, index=False)
    
    logger.info(f"Feature engineering completed. Enhanced dataset saved to {output_path}")
    logger.info(f"Total features: {len(features_df.columns)}")
    
    # Print feature summary
    logger.info("\nFeature Summary:")
    logger.info(f"Linguistic features: {len([f for f in features_df.columns if 'count' in f])}")
    logger.info(f"Scientific features: {len([f for f in features_df.columns if 'scientific' in f])}")
    logger.info(f"Context features: {len([f for f in features_df.columns if 'section' in f or 'context' in f])}")
    logger.info(f"Sentiment features: {len([f for f in features_df.columns if 'sentiment' in f])}")
    logger.info(f"Structural features: {len([f for f in features_df.columns if 'structural' in f or 'paragraph' in f])}")

if __name__ == "__main__":
    main() 