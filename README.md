# Gender Inclusive Language Generation - LT-EDI@ACL 2026

## Task 2: Counter Narrative Generation

This repository contains our submission for the **Gender Inclusive Language Generation Shared Task** at **LT-EDI@ACL 2026**.

### 📋 Task Description

Generate empathetic and persuasive counter-narratives for gender-biased statements.

**Example:**
- **Input (Biased):** "Women are not good at leadership"
- **Output (Counter-narrative):** "People of all gender identities are capable of leadership"

---

## 🏗️ Approach

We fine-tune **Flan-T5-base** on the provided counterfactual sentence pairs using instruction-tuning.

### Model Architecture
- **Base Model:** google/flan-t5-base (250M parameters)
- **Framework:** Hugging Face Transformers
- **Training:** Seq2Seq fine-tuning with instruction prompts

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Epochs | 7 |
| Max Target Length | 256 |
| Label Smoothing | 0.1 |

---

## 📁 Repository Structure

```
├── part6_data_loading.py       # Data loading and preprocessing
├── part8_task_b_training.py    # Training pipeline for Task 2
├── part10_submission.py        # Inference and submission generation
├── requirements.txt            # Dependencies
├── submissions/
│   └── Pranav_Task2_English.csv  # Our submission file
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Download the dataset from Codabench and place in `./data/` directory.

### 3. Train Model
```bash
python part6_data_loading.py
python part8_task_b_training.py
```

### 4. Generate Predictions
```bash
python part10_submission.py
```

---

## 📦 Requirements

```
transformers>=4.30.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
sentencepiece>=0.1.99
accelerate>=0.20.0
```


---

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{pranav-ltedi-2026,
    title = "Pranav@LTEDI 2026: Fine-tuned Flan-T5 for Gender Bias Counter Narrative Generation",
    author = "Pranav Vachharajani",
    booktitle = "Proceedings of the Sixth Workshop on Language Technology for Equality, Diversity, Inclusion",
    month = "July",
    year = "2026",
    publisher = "Association for Computational Linguistics"
}
```

Also cite the shared task:

```bibtex
@inproceedings{gender-inclusive-ltedi-acl-2026,
    title = "Insights from Multilingual Gender Inclusive Language Generation Shared Task",
    author = "Bharathi Raja Chakravarthi and others",
    booktitle = "Proceedings of the Sixth Workshop on Language Technology for Equality, Diversity, Inclusion",
    month = "July",
    year = "2026",
    publisher = "Association for Computational Linguistics"
}
```

---

## 📧 Contact

- **Author:** Pranav Vachharajani


---

## 🙏 Acknowledgments

We thank the LT-EDI@ACL 2026 organizers for creating this shared task.

---

## 📄 License

This project is licensed under the MIT License.
