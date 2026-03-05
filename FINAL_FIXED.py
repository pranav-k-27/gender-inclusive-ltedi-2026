"""
================================================================================
GENDER INCLUSIVE LANGUAGE TASK - PART 8 (PROPERLY FIXED)
Task B Training - Uses ACTUAL data from Part 6
================================================================================

CRITICAL: Run Part 6 first! This script uses:
  - task_b_prepared (your actual training data)
  - test_b_prepared (test data)
================================================================================
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PART 8: TASK B TRAINING - COUNTERFACTUAL GENERATION")
print("=" * 70)

# ============================================================================
# VERIFY DATA FROM PART 6
# ============================================================================
print("\n[1] VERIFYING DATA FROM PART 6")
print("-" * 50)

# Check if data exists from Part 6
try:
    data_count = len(task_b_prepared)
    print(f"  ✓ task_b_prepared found: {data_count} samples")
    
    # Verify it has required columns
    required_cols = ['input', 'output', 'input_with_instruction']
    for col in required_cols:
        if col in task_b_prepared.columns:
            print(f"    ✓ Column '{col}' exists")
        else:
            print(f"    ✗ Column '{col}' MISSING!")
    
    TRAINING_DATA = task_b_prepared.copy()
    
except NameError:
    print("  ✗ task_b_prepared NOT FOUND!")
    print("  ✗ Please run Part 6 first!")
    print("\n  Stopping execution...")
    raise RuntimeError("Part 6 must be run first to load data!")

# Show data sample
print(f"\n  Sample data:")
print(f"    Input: {TRAINING_DATA['input'].iloc[0][:60]}...")
print(f"    Output: {TRAINING_DATA['output'].iloc[0][:60]}...")

# ============================================================================
# CONFIGURATION
# ============================================================================
print("\n[2] CONFIGURATION")
print("-" * 50)

class ConfigTaskB:
    # Model
    MODEL_NAME = "google/flan-t5-base"
    
    # Training - optimized for generative task
    MAX_INPUT_LENGTH = 128
    MAX_TARGET_LENGTH = 256  # Longer for counter-narratives
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5  # Slightly lower for generative
    NUM_EPOCHS = 7  # More epochs for creative task
    WARMUP_STEPS = 150
    WEIGHT_DECAY = 0.01
    
    # Paths
    OUTPUT_DIR = "./task_b_model"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = ConfigTaskB()

print(f"  Model: {config.MODEL_NAME}")
print(f"  Device: {config.DEVICE}")
print(f"  Batch size: {config.BATCH_SIZE}")
print(f"  Epochs: {config.NUM_EPOCHS}")
print(f"  Max target length: {config.MAX_TARGET_LENGTH}")

# ============================================================================
# DATASET CLASS
# ============================================================================
print("\n[3] DATASET CLASS")
print("-" * 50)

class CounterfactualDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len=128, max_target_len=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        input_text = row['input_with_instruction']
        target_text = str(row['output'])
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

print("  Dataset class defined ✓")

# ============================================================================
# PREPARE TRAIN/VAL SPLIT - USING ACTUAL DATA!
# ============================================================================
print("\n[4] PREPARING TRAIN/VALIDATION SPLIT")
print("-" * 50)

# THIS IS THE KEY FIX - use TRAINING_DATA (from Part 6), not sample_data!
train_df, val_df = train_test_split(
    TRAINING_DATA,  # ← Using actual data from Part 6!
    test_size=0.1, 
    random_state=42
)

print(f"  ✓ Training samples: {len(train_df)}")
print(f"  ✓ Validation samples: {len(val_df)}")
print(f"  ✓ Total: {len(train_df) + len(val_df)}")

# ============================================================================
# LOAD MODEL & TOKENIZER
# ============================================================================
print("\n[5] LOADING MODEL & TOKENIZER")
print("-" * 50)

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
model = model.to(config.DEVICE)

print(f"  Model: {config.MODEL_NAME}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# CREATE DATASETS
# ============================================================================
print("\n[6] CREATING PYTORCH DATASETS")
print("-" * 50)

train_dataset = CounterfactualDataset(
    train_df, tokenizer,
    config.MAX_INPUT_LENGTH,
    config.MAX_TARGET_LENGTH
)

val_dataset = CounterfactualDataset(
    val_df, tokenizer,
    config.MAX_INPUT_LENGTH,
    config.MAX_TARGET_LENGTH
)

print(f"  Train dataset: {len(train_dataset)} samples")
print(f"  Val dataset: {len(val_dataset)} samples")

# Verify a sample
sample = train_dataset[0]
decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
print(f"\n  Sample check:")
print(f"    Decoded input: {decoded_input[:70]}...")

# ============================================================================
# TRAINING SETUP
# ============================================================================
print("\n[7] TRAINING SETUP")
print("-" * 50)

training_args = Seq2SeqTrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.NUM_EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    warmup_steps=config.WARMUP_STEPS,
    gradient_accumulation_steps=2,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True,
    generation_max_length=config.MAX_TARGET_LENGTH,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    label_smoothing_factor=0.1,
    report_to="none",
    seed=42
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("  Trainer configured ✓")

# ============================================================================
# TRAINING - UNCOMMENT TO RUN!
# ============================================================================
print("\n[8] TRAINING")
print("-" * 50)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRAINING CONFIGURATION:                                              ║
║                                                                       ║
║  • Training samples: {len(train_df):<10}                                     ║
║  • Validation samples: {len(val_df):<10}                                   ║
║  • Epochs: {config.NUM_EPOCHS:<10}                                              ║
║  • Batch size: {config.BATCH_SIZE:<10}                                          ║
║  • Estimated time: 20-40 minutes on GPU                               ║
║                                                                       ║
║  UNCOMMENT THE LINES BELOW TO START TRAINING!                         ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ===================== UNCOMMENT TO TRAIN =====================
print("Starting training...")
trainer.train()
print("Training complete!")

# Save the model
trainer.save_model(f"{config.OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/final")
print(f"Model saved to {config.OUTPUT_DIR}/final")
# ==============================================================

# ============================================================================
# TEST TRAINED MODEL
# ============================================================================
print("\n[9] TESTING TRAINED MODEL")
print("-" * 50)

def generate_counterfactual(text, model, tokenizer, device):
    """Generate counter-narrative for a biased statement"""
    model.eval()
    instruction = "Generate an empathetic and persuasive counter-narrative for this biased statement: "
    input_text = f"{instruction}{text}"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
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
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test examples
test_examples = [
    "Women are not good at leadership",
    "Men should not show emotions",
    "Girls are weak in mathematics",
    "Boys are naturally aggressive"
]

print("Testing trained model:\n")
for text in test_examples:
    result = generate_counterfactual(text, model, tokenizer, config.DEVICE)
    print(f"  Biased:  {text}")
    print(f"  Counter: {result}")
    print()

# ============================================================================
# PREDICTION FUNCTION FOR SUBMISSION
# ============================================================================
print("\n[10] PREDICTION FUNCTION FOR SUBMISSION")
print("-" * 50)

def generate_predictions_task_b(test_df, model, tokenizer, device, batch_size=4):
    """Generate predictions for test data"""
    model.eval()
    predictions = []
    
    for i in range(0, len(test_df), batch_size):
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
    
    return predictions

print("  Prediction function defined ✓")

print("\n" + "=" * 70)
print("PART 8 COMPLETE!")
print("=" * 70)
print(f"""
TRAINING SUMMARY:
  • Trained on: {len(train_df)} samples
  • Model saved to: {config.OUTPUT_DIR}/final

NEXT STEPS:
  1. Run Part 10 to generate final predictions and create submission
""")
