# ECG Arrhythmia Classification — CS4082 Project

> **Classifying short single-lead ECG recordings into 4 cardiac rhythm categories using machine learning on the PhysioNet Challenge 2017 dataset.**

---

## 📂 Project Structure

```
project-folder/
│
├── ECG_CS4082_Full_Project.ipynb
├── RandomizedSearchCV/
│   ├── RF.py
│   └── xgb.py
├── requirements.txt
└── data/
    ├── training2017.zip
    └── REFERENCE-v3.csv
```

---

## 📌 Project Overview

This project implements an end-to-end machine learning pipeline for ECG arrhythmia classification as part of the CS4082 coursework.

The goal is to classify short, single-lead ECG recordings into four rhythm categories. The workflow includes data exploration, preprocessing, feature extraction, model training, hyperparameter tuning, and evaluation.

---

## 🫀 Classification Target

| Label | Class               | Description                           |
| ----- | ------------------- | ------------------------------------- |
| `N`   | Normal              | Normal sinus rhythm                   |
| `A`   | Atrial Fibrillation | Irregular rhythm, no distinct P-waves |
| `O`   | Other               | Other abnormal rhythms                |
| `~`   | Noise               | Signal too noisy to classify          |

---

## 📊 Dataset

**Source:** https://physionet.org/content/challenge-2017/1.0.0/

The dataset contains ~8,500 ECG recordings (9–61 seconds at 300 Hz) annotated by experts. It is highly imbalanced, with the Normal class dominating.

| Class        | Approx. Count |
| ------------ | ------------- |
| Normal (`N`) | ~5,050        |
| AF (`A`)     | ~738          |
| Other (`O`)  | ~2,496        |
| Noise (`~`)  | ~284          |

Place the following files in the `data/` folder:

* `training2017.zip`
* `REFERENCE-v3.csv`

---

## ⚙️ Pipeline Overview

```
Data Loading → EDA → Preprocessing → Model Training  
→ Hyperparameter Tuning → SMOTE → Feature Selection → Evaluation
```

Key steps include:

* Signal resampling and scaling
* Handling class imbalance using SMOTE
* Training multiple models
* Hyperparameter tuning for Random Forest and XGBoost

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using `RandomizedSearchCV`.

Due to runtime limitations in Google Colab, tuning was executed separately using Python scripts:

* `RF.py` → Random Forest tuning
* `xgb.py` → XGBoost tuning

The best parameters obtained were applied in the main notebook.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the notebook:

```bash
jupyter notebook FTAECG_CS4082_Full_Project.ipynb
```

Run hyperparameter tuning scripts:

```bash
python RandomizedSearchCV/RF.py
python RandomizedSearchCV/xgb.py
```

---

## 📈 Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve

---

## 🔍 Key Insights

* Class imbalance is a major challenge in ECG classification
* XGBoost achieved the best overall performance
* Hyperparameter tuning improved model results

---

## 📝 Notes

Hyperparameter tuning was performed separately due to computational limits in Google Colab.

---
## 📦 Large Files

This project uses Git LFS to store large dataset files (e.g., `.zip`).
Make sure Git LFS is installed to download the dataset correctly.

---

## 👤 Author

Aya Mohammed - 
Afrah Bashaddadah - 
Afnan Kamel

CS4082 Project
