# Breast Cancer Prediction using Logistic Regression

A machine learning project that classifies breast tumors as **benign** or **malignant** from cell-level diagnostic measurements, using Logistic Regression in Python (scikit-learn).

> **Disclaimer:** This project is for learning purposes only. It is not a medical diagnostic tool.

---

## Overview

This is a binary classification project built to practise the end-to-end machine learning workflow:

1. Load and inspect the dataset
2. Split it into training and test sets
3. Train a Logistic Regression classifier
4. Evaluate it with accuracy and a confusion matrix
5. Improve the setup with feature scaling and 10-fold cross-validation

## Dataset

- **Name:** Breast Cancer Wisconsin (Original) dataset (UCI Machine Learning Repository)
- **Samples:** 683 rows
- **Columns:** 11 (a sample ID column, 9 cell-measurement features, and the class label)
- **Target (`Class`):**
  - `2` = Benign
  - `4` = Malignant
- **Features** include Clump Thickness, Uniformity of Cell Size/Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli, and Mitoses, each scored on a 1-10 scale.

The CSV file is expected as `breast_cancer.csv`. Place it in the project folder, or upload it to Colab and update the path in the notebook.

## Tech Stack

- Python
- NumPy, Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook / Google Colab

## Methodology

| Step | Details |
|------|---------|
| Data loading | Read the CSV with Pandas; inspected with `head()` and `tail()` |
| Features / target | All columns except the last as `X`; the last column (`Class`) as `y` |
| Train-test split | 80% train / 20% test (`test_size=0.2`, `random_state=0`) |
| Baseline model | `LogisticRegression(random_state=0)` trained on the raw features |
| Evaluation | Accuracy score and confusion matrix on the test set |
| Improved setup | scikit-learn pipeline: `StandardScaler` + `LogisticRegression(max_iter=1000)` |
| Validation | 10-fold cross-validation on the training set |

## Results

### Baseline model (hold-out test set, 137 samples)

| Metric | Value |
|--------|-------|
| Accuracy | **91.24%** |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted Benign (2) | Predicted Malignant (4) |
|--|:--:|:--:|
| **Actual Benign (2)** | 81 | 6 |
| **Actual Malignant (4)** | 6 | 44 |

Calculated from the confusion matrix, treating malignant as the positive class: precision = 44/50 = 88%, recall = 44/50 = 88%.

### Scaled pipeline (10-fold cross-validation on the training set)

| Metric | Value |
|--------|-------|
| Mean CV accuracy | **96.70%** |
| Standard deviation | 1.97% |

Without feature scaling, the baseline model raised a `ConvergenceWarning`. Standardising the features and increasing `max_iter` resolved this and gave a more reliable cross-validated estimate.

> Note: the 96.70% figure is a cross-validation score on the training data. The scaled pipeline has not yet been scored on the held-out test set, so the two accuracy numbers above are not a direct like-for-like comparison.

## Project Structure

```
breast-cancer-prediction/
├── breast_cancer.ipynb    # Main notebook
├── breast_cancer.csv      # Dataset
├── requirements.txt       # Dependencies
└── README.md
```

## How to Run

### Option 1: Google Colab
1. Upload `breast_cancer.ipynb` to Colab.
2. Upload `breast_cancer.csv` to the Colab session (`/content/`).
3. Run all cells.

### Option 2: Locally
```bash
git clone https://github.com/shaikmofficial-ai/breast-cancer-prediction.git
cd breast-cancer-prediction
pip install -r requirements.txt
jupyter notebook breast_cancer.ipynb
```
If running locally, change the CSV path in the notebook from `/content/breast_cancer.csv` to `breast_cancer.csv`.

### requirements.txt
```
numpy
pandas
matplotlib
scikit-learn
jupyter
```

## Future Improvements

- Remove the sample ID column from the features (it is not a medical measurement and should not be used for prediction)
- Check for and remove duplicate rows
- Evaluate the scaled pipeline on the held-out test set and report precision, recall, and F1-score
- Add visualisations (class distribution, feature correlations, confusion matrix heatmap)
- Compare with other classifiers (KNN, Decision Tree, SVM, Random Forest)
- Tune hyperparameters with `GridSearchCV`
- Wrap the model in a simple web app

## What I Learned

- Building a classification workflow with scikit-learn
- Why feature scaling matters for Logistic Regression
- Reading a confusion matrix and understanding the difference between accuracy and precision/recall
- Using cross-validation for a more reliable performance estimate

## Author

**Shaik Abdulla M**
B.Tech Computer Science and Engineering, Dr. M.G.R. Educational and Research Institute, Chennai
[LinkedIn](https://www.linkedin.com/in/shaik-abdulla-449923380) | [GitHub](https://github.com/shaikmofficial-ai)
