"""
================================================================================
GENDER INCLUSIVE LANGUAGE TASK - PART 10 (FIXED)
Complete Submission Pipeline
================================================================================

IMPORTANT: Run this AFTER training models in Part 7 and Part 8!

This script:
1. Loads your TRAINED models (or pretrained if not found)
2. Generates predictions on test data
3. Creates properly formatted submission files
================================================================================
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import zipfile
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PART 10: COMPLETE SUBMISSION PIPELINE")
print("=" * 70)

# ============================================================================
# CHECK: Did Part 6 run? (Need test data)
# ============================================================================
print("\n[0] CHECKING DATA FROM PART 6")
print("-" * 50)

try:
    print(f"  ✓ test_a_prepared: {len(test_a_prepared)} samples")
    print(f"  ✓ test_b_prepared: {len(test_b_prepared)} samples")
    DATA_LOADED = True
except NameError:
    print("  ✗ Data not found! Creating sample test data...")
    DATA_LOADED = False
    
    # Fallback
    test_a_prepared = pd.DataFrame({
        'id': range(1, 6),
        'input': [
            "The nurse enters the hospital ward.",
            "A teacher starts the first day of class.",
            "The fireman saved the child.",
            "Each salesman must meet his quota.",
            "The chairman called the meeting."
        ],
        'input_with_instruction': [
            "Convert to gender-inclusive language: The nurse enters the hospital ward.",
            "Convert to gender-inclusive language: A teacher starts the first day of class.",
            "Convert to gender-inclusive language: The fireman saved the child.",
            "Convert to gender-inclusive language: Each salesman must meet his quota.",
            "Convert to gender-inclusive language: The chairman called the meeting."
        ]
    })
    
    test_b_prepared = pd.DataFrame({
        'id': range(1, 6),
        'input': [
            "Women are not good at leadership",
            "Men should not show emotions",
            "Girls are weak in mathematics",
            "Boys are naturally aggressive",
            "Women belong only in the kitchen"
        ],
        'input_with_instruction': [
            "Generate an empathetic and persuasive counter-narrative for this biased statement: Women are not good at leadership",
            "Generate an empathetic and persuasive counter-narrative for this biased statement: Men should not show emotions",
            "Generate an empathetic and persuasive counter-narrative for this biased statement: Girls are weak in mathematics",
            "Generate an empathetic and persuasive counter-narrative for this biased statement: Boys are naturally aggressive",
            "Generate an empathetic and persuasive counter-narrative for this biased statement: Women belong only in the kitchen"
        ]
    })

# ============================================================================
# CONFIGURATION
# ============================================================================
print("\n[1] CONFIGURATION")
print("-" * 50)

class SubmissionConfig:
    # Team info - CHANGE THIS!
    TEAM_NAME = "MyTeam"
    
    # Model paths (from training)
    TASK_A_MODEL_PATH = "./task_a_model/final"
    TASK_B_MODEL_PATH = "./task_b_model/final"
    
    # Fallback to pretrained if trained model not found
    PRETRAINED_MODEL = "google/flan-t5-base"
    
    # Output
    OUTPUT_DIR = "./submissions"
    
    # Generation
    MAX_LENGTH_A = 128
    MAX_LENGTH_B = 256
    BATCH_SIZE = 8
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = SubmissionConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print(f"Team Name: {config.TEAM_NAME}")
print(f"Device: {config.DEVICE}")
print(f"Output Directory: {config.OUTPUT_DIR}")

# ============================================================================
# LOAD MODELS
# ============================================================================
print("\n[2] LOADING MODELS")
print("-" * 50)

def load_model_safe(trained_path, pretrained_name, task_name):
    """Load trained model if available, otherwise pretrained"""
    
    if os.path.exists(trained_path):
        print(f"  ✓ Loading TRAINED {task_name} model from: {trained_path}")
        tokenizer = AutoTokenizer.from_pretrained(trained_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(trained_path)
        is_trained = True
    else:
        print(f"  ⚠ Trained model not found at: {trained_path}")
        print(f"    Loading PRETRAINED model: {pretrained_name}")
        print(f"    WARNING: Predictions will be poor without training!")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)
        is_trained = False
    
    return model, tokenizer, is_trained

# Load models
print("\nLoading Task A model...")
model_a, tokenizer_a, is_trained_a = load_model_safe(
    config.TASK_A_MODEL_PATH,
    config.PRETRAINED_MODEL,
    "Task A"
)
model_a = model_a.to(config.DEVICE)
model_a.eval()

print("\nLoading Task B model...")
model_b, tokenizer_b, is_trained_b = load_model_safe(
    config.TASK_B_MODEL_PATH,
    config.PRETRAINED_MODEL,
    "Task B"
)
model_b = model_b.to(config.DEVICE)
model_b.eval()

# Warning if models not trained
if not is_trained_a or not is_trained_b:
    print("\n" + "!" * 70)
    print("WARNING: Using UNTRAINED models!")
    print("Predictions will be poor. Please train models first (Part 7 & 8)")
    print("!" * 70)

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
print("\n[3] PREDICTION FUNCTIONS")
print("-" * 50)

def generate_predictions_task_a(test_df, model, tokenizer, device, batch_size=8):
    """Generate Task A predictions"""
    model.eval()
    predictions = []
    
    total = len(test_df)
    print(f"  Generating {total} Task A predictions...")
    
    for i in range(0, total, batch_size):
        batch = test_df['input_with_instruction'].iloc[i:i+batch_size].tolist()
        
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)
        
        if (i + batch_size) % 50 == 0 or (i + batch_size) >= total:
            print(f"    Progress: {min(i+batch_size, total)}/{total}")
    
    return predictions

def generate_predictions_task_b(test_df, model, tokenizer, device, batch_size=4):
    """Generate Task B predictions"""
    model.eval()
    predictions = []
    
    total = len(test_df)
    print(f"  Generating {total} Task B predictions...")
    
    for i in range(0, total, batch_size):
        batch = test_df['input_with_instruction'].iloc[i:i+batch_size].tolist()
        
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                min_length=20,
                num_beams=5,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)
        
        if (i + batch_size) % 20 == 0 or (i + batch_size) >= total:
            print(f"    Progress: {min(i+batch_size, total)}/{total}")
    
    return predictions

print("Prediction functions defined ✓")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n[4] GENERATING PREDICTIONS")
print("-" * 50)

print("\nTask A - Gender Inclusive Language:")
predictions_a = generate_predictions_task_a(
    test_a_prepared, model_a, tokenizer_a, config.DEVICE, config.BATCH_SIZE
)

print("\nTask B - Counterfactual Generation:")
predictions_b = generate_predictions_task_b(
    test_b_prepared, model_b, tokenizer_b, config.DEVICE, config.BATCH_SIZE // 2
)

print("\n✓ All predictions generated!")

# ============================================================================
# PREVIEW PREDICTIONS
# ============================================================================
print("\n[5] PREVIEW PREDICTIONS")
print("-" * 50)

print("\n--- Task A Samples ---")
for i in range(min(3, len(predictions_a))):
    print(f"\n  Input:  {test_a_prepared['input'].iloc[i][:60]}...")
    print(f"  Output: {predictions_a[i][:60]}...")

print("\n--- Task B Samples ---")
for i in range(min(3, len(predictions_b))):
    print(f"\n  Input:  {test_b_prepared['input'].iloc[i]}")
    print(f"  Output: {predictions_b[i][:80]}...")

# ============================================================================
# QUALITY CHECK
# ============================================================================
print("\n[6] QUALITY CHECK")
print("-" * 50)

import re

def quick_quality_check(predictions, task_name):
    """Quick quality check on predictions"""
    gendered = {'he', 'she', 'him', 'her', 'his', 'hers', 'fireman', 'policeman', 'chairman', 'businessman'}
    neutral = {'they', 'them', 'their', 'firefighter', 'police officer', 'chairperson', 'everyone', 'person'}
    
    stats = {
        'total': len(predictions),
        'empty': 0,
        'with_gendered': 0,
        'with_neutral': 0,
        'avg_length': 0
    }
    
    total_words = 0
    for pred in predictions:
        words = set(re.findall(r'\b\w+\b', pred.lower()))
        total_words += len(pred.split())
        
        if len(pred.strip()) < 5:
            stats['empty'] += 1
        if words & gendered:
            stats['with_gendered'] += 1
        if words & neutral:
            stats['with_neutral'] += 1
    
    stats['avg_length'] = total_words / len(predictions) if predictions else 0
    
    print(f"\n{task_name}:")
    print(f"  Total: {stats['total']} predictions")
    print(f"  Avg length: {stats['avg_length']:.1f} words")
    print(f"  With gendered terms: {stats['with_gendered']} ({stats['with_gendered']/stats['total']*100:.1f}%)")
    print(f"  With neutral terms: {stats['with_neutral']} ({stats['with_neutral']/stats['total']*100:.1f}%)")
    print(f"  Empty/too short: {stats['empty']}")
    
    # Quality assessment
    gendered_pct = stats['with_gendered'] / stats['total'] * 100
    if gendered_pct < 10:
        print(f"  Quality: ✓ GOOD (low gendered terms)")
    elif gendered_pct < 30:
        print(f"  Quality: ⚠ MODERATE (some gendered terms)")
    else:
        print(f"  Quality: ✗ NEEDS TRAINING (high gendered terms)")
    
    return stats

quick_quality_check(predictions_a, "Task A")
quick_quality_check(predictions_b, "Task B")

# ============================================================================
# CREATE SUBMISSION FILES
# ============================================================================
print("\n[7] CREATING SUBMISSION FILES")
print("-" * 50)

def create_submission(predictions, team_name, language, task_num, output_dir):
    """Create CSV and ZIP submission files"""
    
    # Create CSV
    csv_filename = f"{team_name}_{language}.csv"
    csv_path = os.path.join(output_dir, f"task{task_num}", csv_filename)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    df = pd.DataFrame({'prediction': predictions})
    df.to_csv(csv_path, index=False)
    print(f"  Created: {csv_filename} ({len(predictions)} rows)")
    
    # Create ZIP
    zip_filename = f"{team_name}_Task{task_num}.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, csv_filename)
    
    print(f"  Created: {zip_filename}")
    
    return csv_path, zip_path

print("\nTask A submission:")
csv_a, zip_a = create_submission(
    predictions_a, config.TEAM_NAME, "English", 1, config.OUTPUT_DIR
)

print("\nTask B submission:")
csv_b, zip_b = create_submission(
    predictions_b, config.TEAM_NAME, "English", 2, config.OUTPUT_DIR
)

# ============================================================================
# VERIFY SUBMISSION FILES
# ============================================================================
print("\n[8] VERIFY SUBMISSION FILES")
print("-" * 50)

def verify_submission(csv_path, expected_count):
    """Verify submission file format"""
    df = pd.read_csv(csv_path)
    
    checks = {
        'has_prediction_col': 'prediction' in df.columns,
        'correct_rows': len(df) == expected_count,
        'no_empty': df['prediction'].notna().all(),
        'single_column': len(df.columns) == 1
    }
    
    all_pass = all(checks.values())
    status = "✓ PASS" if all_pass else "✗ FAIL"
    
    print(f"\n{os.path.basename(csv_path)}: {status}")
    print(f"  Has 'prediction' column: {checks['has_prediction_col']}")
    print(f"  Correct row count ({expected_count}): {checks['correct_rows']}")
    print(f"  No empty predictions: {checks['no_empty']}")
    print(f"  Single column only: {checks['single_column']}")
    
    return all_pass

verify_submission(csv_a, len(test_a_prepared))
verify_submission(csv_b, len(test_b_prepared))

# ============================================================================
# SUBMISSION SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUBMISSION FILES READY")
print("=" * 70)

print(f"""
Output Directory: {config.OUTPUT_DIR}

Files Created:
  📦 {os.path.basename(zip_a)} - Task A (Gender Inclusive)
  📦 {os.path.basename(zip_b)} - Task B (Counterfactual)

SUBMISSION CHECKLIST:
  ☐ Review predictions in CSV files
  ☐ Verify quality using Part 9 evaluation
  ☐ Submit via Google Form (NOT Codabench directly!)
  ☐ Check submission deadline

IMPORTANT REMINDERS:
  - Submit ZIP files, not CSV files
  - Use Google Form provided by organizers
  - Keep a backup of your submission
""")

if not is_trained_a or not is_trained_b:
    print("""
⚠️  WARNING: You are using UNTRAINED models!
    Your predictions will likely score poorly.
    Please run training in Part 7 and Part 8 first!
""")

print("=" * 70)
print("Good luck with your submission! 🎉")
print("=" * 70)
