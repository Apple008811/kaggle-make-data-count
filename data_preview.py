#!/usr/bin/env python3
"""
Data Preview Script for Kaggle Make Data Count Competition
Shows the structure and content of training and test data
"""

import pandas as pd
import os
import xml.etree.ElementTree as ET
from collections import Counter

def load_xml_files(xml_dir, max_files=5):
    """Load XML files and extract text content"""
    articles = []
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')][:max_files]
    
    for filename in xml_files:
        filepath = os.path.join(xml_dir, filename)
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Extract text content (simplified)
            text_content = ""
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_content += elem.text.strip() + " "
            
            article_id = filename.replace('.xml', '')
            articles.append({
                'article_id': article_id,
                'text': text_content[:500] + "..." if len(text_content) > 500 else text_content,
                'filename': filename
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return pd.DataFrame(articles)

def preview_training_data():
    """Preview training data"""
    print("=" * 60)
    print("TRAINING DATA PREVIEW")
    print("=" * 60)
    
    # Load train_labels.csv
    train_labels_path = '/kaggle/input/make-data-count-finding-data-references/train_labels.csv'
    if os.path.exists(train_labels_path):
        train_df = pd.read_csv(train_labels_path)
        print(f"✅ Loaded {len(train_df)} training samples")
        print(f"Columns: {train_df.columns.tolist()}")
        print(f"Shape: {train_df.shape}")
        
        # Show first 10 rows
        print("\n📋 First 10 rows of train_labels.csv:")
        print(train_df.head(10))
        
        # Show type distribution
        print(f"\n📊 Type distribution:")
        type_counts = train_df['type'].value_counts()
        print(type_counts)
        print(f"Distribution percentages:")
        for type_name, count in type_counts.items():
            percentage = (count / len(train_df)) * 100
            print(f"  {type_name}: {count} ({percentage:.1f}%)")
        
        # Show unique article_ids
        unique_articles = train_df['article_id'].nunique()
        print(f"\n📚 Unique articles: {unique_articles}")
        
        # Show dataset_id examples
        print(f"\n🔗 Dataset ID examples:")
        sample_datasets = train_df['dataset_id'].head(10).tolist()
        for i, dataset in enumerate(sample_datasets, 1):
            print(f"  {i}. {dataset}")
        
        # Show some statistics
        print(f"\n📈 Statistics:")
        print(f"  Total predictions: {len(train_df)}")
        print(f"  Average datasets per article: {len(train_df) / unique_articles:.1f}")
        
        # Check for DOIs vs Accession IDs
        doi_count = train_df['dataset_id'].str.contains('doi.org').sum()
        accession_count = len(train_df) - doi_count
        print(f"  DOI-based citations: {doi_count} ({doi_count/len(train_df)*100:.1f}%)")
        print(f"  Accession ID citations: {accession_count} ({accession_count/len(train_df)*100:.1f}%)")
        
    else:
        print("❌ train_labels.csv not found")

def preview_test_data():
    """Preview test data"""
    print("\n" + "=" * 60)
    print("TEST DATA PREVIEW")
    print("=" * 60)
    
    # Try to load test.csv first
    test_csv_path = '/kaggle/input/make-data-count-finding-data-references/test.csv'
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        print(f"✅ Loaded {len(test_df)} test samples from CSV")
        print(f"Columns: {test_df.columns.tolist()}")
        print(f"Shape: {test_df.shape}")
        
        print("\n📋 First 10 rows of test.csv:")
        print(test_df.head(10))
        
    else:
        print("📄 test.csv not found, loading from XML files...")
        
        # Load from XML files
        xml_dir = '/kaggle/input/make-data-count-finding-data-references/test/XML'
        if os.path.exists(xml_dir):
            test_df = load_xml_files(xml_dir, max_files=5)
            print(f"✅ Loaded {len(test_df)} test samples from XML")
            print(f"Columns: {test_df.columns.tolist()}")
            print(f"Shape: {test_df.shape}")
            
            print("\n📋 First 5 XML files content preview:")
            for i, row in test_df.iterrows():
                print(f"\n--- Article {i+1}: {row['article_id']} ---")
                print(f"Filename: {row['filename']}")
                print(f"Text preview: {row['text'][:200]}...")
        else:
            print("❌ Test XML directory not found")

def preview_sample_submission():
    """Preview sample submission format"""
    print("\n" + "=" * 60)
    print("SAMPLE SUBMISSION FORMAT")
    print("=" * 60)
    
    sample_path = '/kaggle/input/make-data-count-finding-data-references/sample_submission.csv'
    if os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        print(f"✅ Loaded sample submission")
        print(f"Columns: {sample_df.columns.tolist()}")
        print(f"Shape: {sample_df.shape}")
        
        print("\n📋 Sample submission format:")
        print(sample_df.head(10))
        
        # Show type distribution in sample
        if 'type' in sample_df.columns:
            print(f"\n📊 Sample type distribution:")
            print(sample_df['type'].value_counts())
    else:
        print("❌ sample_submission.csv not found")

def main():
    """Main preview function"""
    print("🔍 KAGGLE MAKE DATA COUNT - DATA PREVIEW")
    print("=" * 60)
    print("This script shows the structure and content of the competition data")
    print("=" * 60)
    
    # Check if we're in Kaggle environment
    if os.path.exists('/kaggle/input'):
        print("✅ Running in Kaggle environment")
        input_dir = '/kaggle/input/make-data-count-finding-data-references'
        if os.path.exists(input_dir):
            print(f"✅ Found competition data at: {input_dir}")
            print(f"📁 Available files/directories:")
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
        else:
            print("❌ Competition data not found")
            return
    else:
        print("⚠️ Not running in Kaggle environment")
        print("This script is designed to run on Kaggle")
        return
    
    # Preview all data types
    preview_training_data()
    preview_test_data()
    preview_sample_submission()
    
    print("\n" + "=" * 60)
    print("DATA PREVIEW COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main() 