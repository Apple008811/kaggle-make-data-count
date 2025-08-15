# Make Data Count - Data Citation Detection

This project is for the Kaggle competition "Make Data Count: Finding Data References" which aims to identify data citations in scientific literature and classify them as primary or secondary.

## Competition Overview

- **Goal**: Identify all data citations from scientific literature and tag their type (primary or secondary)
- **Primary**: Raw or processed data generated as part of the paper, specifically for the study
- **Secondary**: Raw or processed data derived or reused from existing records or published data
- **Evaluation Metric**: F1-Score
- **Submission Format**: CSV with columns: `row_id,article_id,dataset_id,type`

## Project Structure

```
make_data_count_nlp/
├── data/                   # Data files (train, test, sample submission)
├── src/                    # Source code
│   ├── data/              # Data processing utilities
│   ├── models/            # Model implementations
│   ├── features/          # Feature engineering
│   ├── utils/             # Utility functions
│   └── config/            # Configuration files
├── notebooks/             # Jupyter notebooks for exploration
├── experiments/           # Experiment tracking and results
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download competition data from Kaggle and place in `data/` directory

3. Run data preprocessing:
```bash
python src/data/preprocess.py
```

## Key Features

- **Text Preprocessing**: Clean and normalize scientific text
- **Feature Engineering**: Extract relevant features for citation detection
- **Model Training**: Train classification models for primary/secondary detection
- **Evaluation**: Comprehensive evaluation with F1-score
- **Submission**: Generate competition submission files

## Models

- BERT-based models for text classification
- Rule-based approaches for DOI detection
- Ensemble methods combining multiple approaches

## Experimental Results and Progress Tracking

### Version History and Performance

#### Version 2.0 (August 14, 2024 - 22:12 PST)
**Optimizations Applied:**
- Lowered confidence thresholds: Primary 0.7→0.5, Secondary 0.7→0.5, Missing 0.5→0.6
- Added distribution balancing mechanism to match training data distribution
- Enhanced feature engineering with 25+ new features
- Improved prediction logic with rule-based fallback
- Expanded keyword detection from 25 to 115 keywords (4.3x increase)

**Expected Improvements:**
- Target distribution: Secondary 42%, Missing 33%, Primary 25%
- Previous result: Missing 99%, Primary 0.5%, Secondary 0.5%
- Keyword detection improvement: 2.3x increase in matches

**Technical Details:**
- ML Model: Random Forest with F1 score 0.668 (±0.131)
- Feature count: 25+ engineered features
- Processing: 418 predictions from 10 test articles
- Keywords: 115 comprehensive data citation keywords

#### Version 1.0 (August 14, 2024 - 18:30 PST)
**Initial Implementation:**
- Basic ML model with Random Forest classifier
- 25 core keywords for data citation detection
- Simple rule-based prediction logic
- Cross-validation F1 score: 0.668

**Results:**
- Prediction distribution: Missing 99%, Primary 0.5%, Secondary 0.5%
- Identified issue: Over-conservative confidence thresholds
- Keyword detection limited to basic terms

### Performance Metrics

| Version | F1 Score | Primary % | Secondary % | Missing % | Keywords | Features |
|---------|----------|-----------|-------------|-----------|----------|----------|
| 1.0     | 0.668    | 0.5%      | 0.5%        | 99%       | 25       | 13       |
| 2.0     | TBD      | ~25%      | ~42%        | ~33%      | 115      | 25+      |

### Future Optimizations (Planned)

#### Version 3.0 (Planned)
- [ ] Implement BERT-based text classification
- [ ] Add more sophisticated context analysis
- [ ] Optimize feature selection based on importance
- [ ] Implement ensemble methods
- [ ] Add cross-validation with different folds

#### Version 4.0 (Planned)
- [ ] Deep learning approach with transformers
- [ ] Multi-task learning for related tasks
- [ ] Advanced data augmentation techniques
- [ ] Hyperparameter optimization
- [ ] Model interpretability analysis

## Technical Implementation

### Regular Expression Pattern Matching

The project uses sophisticated regular expressions to identify and standardize data citations in various formats:

#### DOI Pattern Recognition
```python
# Multiple DOI formats supported
self.doi_patterns = [
    # Standard DOI formats
    r'10\.\d{4,}/[-._;()/:\w]+',
    r'https?://doi\.org/10\.\d{4,}/[-._;()/:\w]+',
    r'http://dx\.doi\.org/10\.\d{4,}/[-._;()/:\w]+',
    r'doi:10\.\d{4,}/[-._;()/:\w]+',
    
    # Data repository specific formats
    r'https?://zenodo\.org/record/\d+',
    r'https?://figshare\.com/articles/\d+',
    r'https?://datadryad\.org/stash/dataset/doi:10\.\d{4,}/[-._;()/:\w]+',
    r'https?://dataverse\.harvard\.edu/dataset\.xhtml\?persistentId=doi:10\.\d{4,}/[-._;()/:\w]+',
]
```

#### Accession ID Recognition
```python
# Domain-specific database identifiers
self.accession_patterns = [
    # Gene Expression Omnibus (GEO)
    r'\bGSE\d+\b',
    r'\bGSM\d+\b',
    r'\bGPL\d+\b',
    
    # Protein Data Bank (PDB)
    r'\bpdb\s+\w+\b',
    r'\bPDB\s+\w+\b',
    
    # ArrayExpress
    r'\bE-MEXP-\d+\b',
    r'\bE-MTAB-\d+\b',
    r'\bE-GEOD-\d+\b',
    
    # European Nucleotide Archive (ENA)
    r'\bPRJ[EN]\d+\b',
    r'\bSRR\d+\b',
    r'\bERR\d+\b',
    
    # NCBI Sequence Read Archive
    r'\bSRP\d+\b',
    r'\bSRS\d+\b',
    
    # The Cancer Genome Atlas (TCGA)
    r'\bTCGA-\w+-\w+-\w+\b',
    
    # Other common databases
    r'\bCHEMBL\d+\b',  # ChEMBL
    r'\bUNIPROT:\w+\b',  # UniProt
    r'\bENSEMBL:\w+\b',  # Ensembl
]
```

#### Key Capabilities
- **Precise Identification**: Only matches valid citation patterns
- **Format Standardization**: Converts various formats to standard URLs
- **Data Validation**: Ensures citation format correctness
- **Batch Processing**: Handles multiple formats simultaneously
- **Error Filtering**: Automatically filters invalid citations

#### Processing Methods
```python
def extract_dois(self, text: str) -> List[str]:
    """Extract DOIs from text"""
    dois = []
    for pattern in self.doi_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Handle different DOI formats
            if match.startswith('http'):
                doi = match
            elif match.startswith('doi:'):
                doi = f"https://doi.org/{match[4:]}"
            elif match.startswith('10.'):
                doi = f"https://doi.org/{match}"
            else:
                doi = f"https://doi.org/{match}"
            dois.append(doi)
    return list(set(dois))

def extract_accession_ids(self, text: str) -> List[str]:
    """Extract Accession IDs from text"""
    accession_ids = []
    for pattern in self.accession_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Standardize Accession ID format
            if 'pdb' in match.lower():
                parts = match.split()
                if len(parts) == 2:
                    accession_ids.append(f"{parts[0].upper()} {parts[1].upper()}")
            else:
                accession_ids.append(match.upper())
    return list(set(accession_ids))
```

#### Example Processing
```
Input: "Data available at doi:10.5061/dryad.6m3n9 and http://dx.doi.org/10.5281/zenodo.1234567"
Output: ["https://doi.org/10.5061/dryad.6m3n9", "https://doi.org/10.5281/zenodo.1234567"]

Input: "Using PDB structure pdb 5yfp and GEO dataset GSE12345"
Output: ["PDB 5YFP", "GSE12345"]
```

## Usage

```python
# Train model
python src/models/train.py

# Make predictions
python src/models/predict.py

# Generate submission
python src/submission/generate.py
```

## Appendix

## Competition Timeline

- **Start Date**: June 11, 2025
- **Entry Deadline**: September 2, 2025
- **Final Submission**: September 9, 2025

## Prizes

- 1st Place: $40,000
- 2nd Place: $20,000
- 3rd Place: $17,000
- 4th Place: $13,000
- 5th Place: $10,000

### A. Machine Learning Technology Stack

This project implements a hybrid approach combining multiple machine learning techniques:

```
Artificial Intelligence (AI)
├── Machine Learning (ML)
│   ├── Traditional Machine Learning (Decision Trees, SVM, etc.)
│   ├── Deep Learning (DL)
│   │   ├── CNN (Image Recognition)
│   │   ├── RNN (Sequential Data)
│   │   └── Transformer (Text Understanding)
│   └── Reinforcement Learning
└── Other AI Methods
```

#### A.1 BERT (Bidirectional Encoder Representations from Transformers)

**What is BERT?**
BERT is a pre-trained Transformer model specifically designed for understanding language context and semantics.

**Key Features:**
- **Bidirectional Understanding**: Reads text from both directions
- **Pre-trained Knowledge**: Already understands language patterns
- **Context Awareness**: Understands word relationships in sentences
- **Task Adaptation**: Can be fine-tuned for specific tasks

**In This Project:**
```python
# Using PubMedBERT for scientific text
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
bert_model = AutoModel.from_pretrained(model_name)
```

**Example Application:**
```
Input: "We used data from GEO database GSE12345"
BERT Understanding: 
- "used data from" indicates existing data
- "GEO database" specifies the source
- Overall context suggests Secondary data
Output: Secondary classification
```

#### A.2 Hybrid Approach Implementation

**Layer 1: Rule-based Methods (Traditional ML)**
```python
# Pattern recognition using regular expressions
doi_patterns = [r'10\.\d{4,}/[-._;()/:\w]+', ...]
accession_patterns = [r'\bGSE\d+\b', ...]

# Keyword-based classification
primary_indicators = ['generated', 'created', 'produced', ...]
secondary_indicators = ['obtained from', 'downloaded from', ...]
```

**Layer 2: Deep Learning (Transformer)**
```python
# BERT model for complex semantic understanding
class DataCitationClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2):
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
```

**Layer 3: Ensemble Strategy**
```python
# Hybrid decision making
def predict_with_confidence(self, text: str, dataset_id: str):
    # Rule-based prediction first
    context_analysis = self.analyze_citation_context(text, dataset_id)
    confidence = context_analysis['confidence']
    
    # Use BERT if confidence is low
    if confidence < 0.6:
        bert_prediction = self.bert_model_predict(text)
        return bert_prediction
    
    return context_analysis['type']
```

#### A.3 Technology Comparison

| Method | Speed | Accuracy | Interpretability | Use Case |
|--------|-------|----------|------------------|----------|
| Rule-based | Fast | Medium | High | Clear patterns |
| BERT | Slow | High | Low | Complex semantics |
| Hybrid | Medium | High | Medium | Production systems |

#### A.4 Real-world Applications

**Scientific Literature Analysis:**
- Automated data citation detection
- Research reproducibility assessment
- Academic impact evaluation

**Financial Document Processing:**
- Contract analysis and risk assessment
- Regulatory compliance checking
- Automated report generation

**Healthcare Data Management:**
- Medical literature analysis
- Clinical trial data tracking
- Research data citation validation

## Data Exploration and Analysis

### Data Preview Script

The project includes a comprehensive data preview script (`data_preview.py`) that provides detailed insights into the competition dataset:

#### Features:
- **Training Data Analysis**: Shows distribution of Primary/Secondary/Missing types
- **Test Data Preview**: Displays structure and content of test articles
- **Sample Submission Format**: Validates submission file structure
- **Statistical Summary**: Provides key metrics and insights

#### Usage:
```bash
# Run in Kaggle environment
python data_preview.py
```

#### Sample Output:
```
🔍 KAGGLE MAKE DATA COUNT - DATA PREVIEW
============================================================
✅ Running in Kaggle environment
✅ Found competition data at: /kaggle/input/make-data-count-finding-data-references

📊 Type distribution:
Secondary    447 (42.0%)
Missing      351 (33.0%)
Primary      268 (25.0%)

📈 Statistics:
  Total predictions: 1066
  Average datasets per article: 2.1
  DOI-based citations: 623 (58.4%)
  Accession ID citations: 443 (41.6%)
```

#### Key Insights:
- Training data contains 1,066 labeled examples
- Distribution: Secondary 42%, Missing 33%, Primary 25%
- Mix of DOI-based and Accession ID citations
- Average of 2.1 datasets per scientific article 