"""
================================================================================
GENDER INCLUSIVE LANGUAGE TASK - PART 6 (CLEANED)
Data Loading & Preprocessing - USING ACTUAL KAGGLE DATA
================================================================================
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

print("=" * 70)
print("PART 6: DATA LOADING & PREPROCESSING")
print("=" * 70)

# ============================================================================
# SECTION 1: Define Data Paths (KAGGLE PATHS)
# ============================================================================
print("\n[1] SETTING UP DATA PATHS")
print("-" * 50)

# Kaggle input directory
DATA_DIR = Path("/kaggle/input")

# Task A paths
TASK_A_SENTENCE_PAIRS = DATA_DIR / "task-a-train-data/Gender Neutral Sentence Pairs.xlsx - Inclusive Pairs.csv"
TASK_A_WORD_PAIRS = DATA_DIR / "task-a-train-data/Gender Neutral Pairs.xlsx - Sheet1.csv"

# Task B paths  
TASK_B_COUNTERFACTUAL = DATA_DIR / "task-b-train-data/Counterfactual Sentence Pairs.xlsx - CounterFactual Data.csv"

# Test data paths
TEST_TASK_A = DATA_DIR / "test-data/English.xlsx - English.csv"
TEST_TASK_B = DATA_DIR / "test-data/English - CF.xlsx - Sheet1.csv"

print(f"Data directory: {DATA_DIR}")

# Verify files exist
print("\nChecking files:")
for name, path in [
    ("Task A Sentences", TASK_A_SENTENCE_PAIRS),
    ("Task A Words", TASK_A_WORD_PAIRS),
    ("Task B", TASK_B_COUNTERFACTUAL),
    ("Test A", TEST_TASK_A),
    ("Test B", TEST_TASK_B)
]:
    exists = "✓" if path.exists() else "✗ NOT FOUND"
    print(f"  {exists} {name}: {path.name}")

# ============================================================================
# SECTION 2: Text Cleaning Functions
# ============================================================================
print("\n[2] TEXT CLEANING FUNCTIONS")
print("-" * 50)

def fix_encoding(text):
    """Fix common encoding issues in the dataset"""
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Common encoding fixes
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã ': 'à',
        '\xa0': ' ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def clean_text(text):
    """General text cleaning"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = fix_encoding(text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

print("Text cleaning functions defined ✓")

# ============================================================================
# SECTION 3: Data Loading Functions
# ============================================================================
print("\n[3] DATA LOADING FUNCTIONS")
print("-" * 50)

def load_task_a_sentences(filepath):
    """Load Task A sentence pairs"""
    df = pd.read_csv(filepath)
    print(f"  Raw columns: {df.columns.tolist()}")
    
    # Standardize column names
    df = df.rename(columns={
        'non-inclusive': 'input',
        'inclusive': 'output',
        'Category': 'category'
    })
    
    # Clean data
    df = df.dropna(subset=['input', 'output'])
    df['input'] = df['input'].apply(clean_text)
    df['output'] = df['output'].apply(clean_text)
    
    return df

def load_task_a_word_pairs(filepath):
    """Load Task A word pairs as dictionary"""
    df = pd.read_csv(filepath)
    
    word_map = {}
    for _, row in df.iterrows():
        original = str(row['Original Terms']).strip().lower()
        inclusive = str(row['Inclusive Terms']).strip().lower()
        word_map[original] = inclusive
        word_map[original.capitalize()] = inclusive.capitalize()
        word_map[original.upper()] = inclusive.upper()
    
    return word_map, df

def load_task_b_data(filepath):
    """Load Task B counterfactual pairs"""
    df = pd.read_csv(filepath)
    print(f"  Raw columns: {df.columns.tolist()}")
    
    df = df.rename(columns={
        'Biased Sentence': 'input',
        'Counterfactual Sentence': 'output',
        'Category': 'category',
        'Pair ID': 'pair_id'
    })
    
    df = df.dropna(subset=['input', 'output'])
    df['input'] = df['input'].apply(clean_text)
    df['output'] = df['output'].apply(clean_text)
    
    return df

def load_test_a(filepath):
    """Load Task A test data"""
    df = pd.read_csv(filepath)
    print(f"  Raw columns: {df.columns.tolist()}")
    
    df = df.rename(columns={
        'Test Case ID': 'id',
        'Input Prompt': 'input'
    })
    
    df['input'] = df['input'].apply(clean_text)
    return df

def load_test_b(filepath):
    """Load Task B test data"""
    df = pd.read_csv(filepath)
    print(f"  Raw columns: {df.columns.tolist()}")
    
    df = df.rename(columns={
        'Pair ID': 'id',
        'Biased Sentence': 'input'
    })
    
    df['input'] = df['input'].apply(clean_text)
    return df

print("Data loading functions defined ✓")

# ============================================================================
# SECTION 4: LOAD ACTUAL DATA
# ============================================================================
print("\n[4] LOADING ACTUAL DATA")
print("-" * 50)

# Load Task A data
print("\nLoading Task A sentence pairs...")
task_a_sentences = load_task_a_sentences(TASK_A_SENTENCE_PAIRS)
print(f"  Loaded: {len(task_a_sentences)} sentence pairs")

print("\nLoading Task A word pairs...")
word_lookup, task_a_words_df = load_task_a_word_pairs(TASK_A_WORD_PAIRS)
print(f"  Loaded: {len(word_lookup)//3} word pairs (with case variations: {len(word_lookup)})")

# Load Task B data
print("\nLoading Task B counterfactual data...")
task_b_data = load_task_b_data(TASK_B_COUNTERFACTUAL)
print(f"  Loaded: {len(task_b_data)} counterfactual pairs")

# Load Test data
print("\nLoading Test A data...")
test_a = load_test_a(TEST_TASK_A)
print(f"  Loaded: {len(test_a)} test samples")

print("\nLoading Test B data...")
test_b = load_test_b(TEST_TASK_B)
print(f"  Loaded: {len(test_b)} test samples")

# ============================================================================
# SECTION 5: Prepare Data with Instructions
# ============================================================================
print("\n[5] PREPARING DATA WITH INSTRUCTIONS")
print("-" * 50)

# Task A instruction
TASK_A_INSTRUCTION = "Convert to gender-inclusive language: "

# Task B instruction
TASK_B_INSTRUCTION = "Generate an empathetic and persuasive counter-narrative for this biased statement: "

def prepare_for_training(df, instruction):
    """Add instruction prefix to inputs"""
    df = df.copy()
    df['input_with_instruction'] = df['input'].apply(lambda x: f"{instruction}{x}")
    return df

# Prepare training data
task_a_prepared = prepare_for_training(task_a_sentences, TASK_A_INSTRUCTION)
task_b_prepared = prepare_for_training(task_b_data, TASK_B_INSTRUCTION)

# Prepare test data
test_a_prepared = prepare_for_training(test_a, TASK_A_INSTRUCTION)
test_b_prepared = prepare_for_training(test_b, TASK_B_INSTRUCTION)

print("Data prepared with instructions ✓")

# ============================================================================
# SECTION 6: Data Preview
# ============================================================================
print("\n[6] DATA PREVIEW")
print("-" * 50)

print("\n--- TASK A Sample ---")
print(f"Input: {task_a_prepared['input'].iloc[0]}")
print(f"With instruction: {task_a_prepared['input_with_instruction'].iloc[0]}")
print(f"Target output: {task_a_prepared['output'].iloc[0]}")

print("\n--- TASK B Sample ---")
print(f"Input: {task_b_prepared['input'].iloc[0]}")
print(f"With instruction: {task_b_prepared['input_with_instruction'].iloc[0]}")
print(f"Target output: {task_b_prepared['output'].iloc[0]}")

print("\n--- TEST A Sample ---")
print(f"ID: {test_a_prepared['id'].iloc[0]}")
print(f"Input: {test_a_prepared['input'].iloc[0]}")
print(f"With instruction: {test_a_prepared['input_with_instruction'].iloc[0]}")

print("\n--- TEST B Sample ---")
print(f"ID: {test_b_prepared['id'].iloc[0]}")
print(f"Input: {test_b_prepared['input'].iloc[0]}")
print(f"With instruction: {test_b_prepared['input_with_instruction'].iloc[0]}")

# ============================================================================
# SECTION 7: Data Statistics
# ============================================================================
print("\n[7] DATA STATISTICS")
print("-" * 50)

def print_stats(df, name):
    print(f"\n{name}:")
    print(f"  Total samples: {len(df)}")
    if 'input' in df.columns:
        lengths = df['input'].str.split().str.len()
        print(f"  Input length (words): min={lengths.min()}, max={lengths.max()}, avg={lengths.mean():.1f}")
    if 'output' in df.columns:
        lengths = df['output'].str.split().str.len()
        print(f"  Output length (words): min={lengths.min()}, max={lengths.max()}, avg={lengths.mean():.1f}")

print_stats(task_a_prepared, "Task A Training Data")
print_stats(task_b_prepared, "Task B Training Data")
print_stats(test_a_prepared, "Test A Data")
print_stats(test_b_prepared, "Test B Data")

# ============================================================================
# SECTION 8: Export Variables for Next Parts
# ============================================================================
print("\n[8] VARIABLES READY FOR TRAINING")
print("-" * 50)

print("""
The following variables are now ready to use in Part 7 & 8:

TRAINING DATA:
  - task_a_prepared : DataFrame with 'input', 'output', 'input_with_instruction'
  - task_b_prepared : DataFrame with 'input', 'output', 'input_with_instruction'
  - word_lookup     : Dictionary for word-level replacements

TEST DATA:
  - test_a_prepared : DataFrame with 'id', 'input', 'input_with_instruction'
  - test_b_prepared : DataFrame with 'id', 'input', 'input_with_instruction'

INSTRUCTIONS:
  - TASK_A_INSTRUCTION : "Convert to gender-inclusive language: "
  - TASK_B_INSTRUCTION : "Generate an empathetic and persuasive counter-narrative..."
""")

print("\n" + "=" * 70)
print("Part 6 Complete! Data loaded and ready for training.")
print("=" * 70)
print(f"""
SUMMARY:
  Task A Training: {len(task_a_prepared)} samples
  Task B Training: {len(task_b_prepared)} samples
  Test A: {len(test_a_prepared)} samples
  Test B: {len(test_b_prepared)} samples
  Word Pairs: {len(word_lookup)//3} pairs

Next: Run Part 7 for Task A training, Part 8 for Task B training
""")
