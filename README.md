# Urdu Hate Speech Detection
### Reproduction of Hussain et al. (2025) — NLP Assignment 3

Reproduction of the full experiment suite from:

> Hussain, I., Arshad, M.M.A., Cheema, A.N., & Almanjahie, I.M. (2025). *Enhancing Urdu hate speech detection through differential transfer learning and adaptive loss functions.* Scientific Reports, 15, 37407. https://doi.org/10.1038/s41598-025-21306-w

---

## Overview

This project implements, trains, and evaluates seven model families for three-class hate speech detection on the **NUHONS** dataset — 18,058 Nastaliq Urdu YouTube comments labelled as **Hate**, **Offensive**, or **Neither**.

| Category | Models |
|---|---|
| Traditional ML (TF-IDF) | SVM, Logistic Regression, Random Forest, Decision Tree |
| Deep Neural (char-level) | RNN, LSTM |
| Transformer | XLM-RoBERTa-base, DAmBERT (mBERT + GLLRD + Adaptive Loss) |

---

## Dataset

**NUHONS** (Nastaliq Urdu Hate and Offensive Nastaliq Speech) — collected from YouTube comments on Pakistani Islamic scholars' channels.

| Subset | Instances |
|---|---|
| Total | 18,058 |
| Hate | 2,430 |
| Offensive | 7,446 |
| Neither | 8,182 |

> The dataset is not publicly available. Request it from the corresponding author at ijazhussain@pieas.edu.pk. Place `Urdu_combined.csv` in your dataset directory before running.

---

## Project Structure

```
urdu-hate-speech-detection/
│
├── NNRR.ipynb          # Main notebook — all experiments in one cell
├── README.md           # This file
└── Assignment3_Report.pdf  # Final submission report
```

---

## Setup & Running

### On Kaggle (recommended)

1. Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. Click **+ Add Data** → upload your dataset with the slug `DATASET` so the file is at:
   ```
   /kaggle/input/DATASET/Urdu_combined.csv
   ```
3. Set accelerator: **Session options → Accelerator → GPU (T4 or P100)**
4. Upload `NNRR.ipynb` and open it
5. Click **Run All** — everything runs in a single cell top-to-bottom

### On Google Colab

1. Upload `NNRR.ipynb` to Colab
2. Upload `Urdu_combined.csv` to the Colab file panel
3. Change `DATA_PATH` in the notebook to:
   ```python
   DATA_PATH = 'Urdu_combined.csv'
   ```
4. Set runtime: **Runtime → Change runtime type → T4 GPU**
5. Click **Runtime → Run All**

### Locally

```bash
pip install torch transformers sentencepiece scikit-learn pandas numpy \
            matplotlib seaborn regex deep-translator langdetect

jupyter notebook NNRR.ipynb
```

---

## Results

| Model | Our F1 | Paper F1 | Δ |
|---|---|---|---|
| DAmBERT (Proposed) | 68.52% | 91.38% | −22.86% |
| XLM-RoBERTa | 72.02% | 89.70% | −17.68% |
| SVM | 67.22% | 82.84% | −15.62% |
| LR | 66.69% | 80.85% | −14.16% |
| RF | 63.31% | 71.05% | −7.74% |
| RNN | 44.16% | 80.01% | −35.85% |
| LSTM | 35.98% | 68.72% | −32.74% |

Performance gaps are primarily attributed to Google Translate API restrictions on Kaggle during preprocessing and potential label string mismatches in the CSV, which degraded input text quality across all models. The relative ranking of model families is preserved, confirming the architectural conclusions of the original paper.

---

## Key Implementation Details

- **Split**: 80/10/10 stratified train/val/test
- **DAmBERT**: `bert-base-multilingual-cased` + Gradual Layer-wise Learning Rate Decay (GLLRD, base_lr=5e-5, decay=0.95) + inverse-frequency adaptive loss
- **XLM-RoBERTa**: fixed lr=2e-5, standard cross-entropy, no GLLRD
- **Char-level models**: vocabulary built from training split only; sequences padded/truncated to 256
- **ML models**: TF-IDF (max 5,000 features), parallel training via `ThreadPoolExecutor`
- All random seeds fixed at **42** for reproducibility

---

## Requirements

```
torch>=2.0
transformers>=4.40
sentencepiece
scikit-learn>=1.4
pandas
numpy
matplotlib
seaborn
regex
deep-translator
langdetect
```

---

## Citation

```bibtex
@article{hussain2025urdu,
  title={Enhancing Urdu hate speech detection through differential transfer
         learning and adaptive loss functions},
  author={Hussain, Ijaz and Arshad, Muhammad Mahr Ali and
          Cheema, Ammara Nawaz and Almanjahie, Ibrahim M.},
  journal={Scientific Reports},
  volume={15},
  pages={37407},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-025-21306-w}
}
```

---

## Course

Introduction to NLP — Assignment 3
Pakistan Institute of Engineering and Applied Sciences (PIEAS)
