# 🍷 Feature Selection Lab – Red Wine Quality Prediction

A hands-on machine learning lab that demonstrates different **Feature Selection techniques** to predict whether a red wine is of good or bad quality based on its chemical properties.

---

## 🎯 Objective

The goal of this lab is to explore and compare different feature selection methods to find the most relevant features for predicting wine quality — reducing model complexity while maintaining or improving performance.

---

## 📂 Project Structure

```
Feature Selection Lab/
├── data/
│   └── winequality-red.csv       # Red wine dataset (UCI)
├── wine_feature_selection.py     # Main lab script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository – Wine Quality Dataset
- **Type:** Red Wine samples from Portugal (Vinho Verde)
- **Size:** 1,599 samples × 11 features
- **Target:** `quality` score binarized into:
  - `0` = Low quality (score ≤ 5)
  - `1` = High quality (score ≥ 6)

### Features:
| Feature | Description |
|---|---|
| fixed acidity | Amount of fixed acids |
| volatile acidity | Amount of acetic acid |
| citric acid | Freshness and flavor |
| residual sugar | Sugar remaining after fermentation |
| chlorides | Amount of salt |
| free sulfur dioxide | Free SO2 in wine |
| total sulfur dioxide | Total SO2 in wine |
| density | Density of the wine |
| pH | Acidity level |
| sulphates | Antimicrobial additive |
| alcohol | Alcohol percentage |

---

## 🔬 Feature Selection Methods Covered

### 📌 Filter Methods
Statistical techniques that score features **independently of any model**.

| Method | Description |
|---|---|
| Pearson Correlation | Selects features with \|r\| > 0.1 correlation with target |
| Inter-feature Correlation | Removes redundant highly correlated features |
| ANOVA F-test (SelectKBest) | Picks top 8 features by F-score |

### 📌 Wrapper Methods
Use a model to **evaluate subsets of features**.

| Method | Description |
|---|---|
| RFE (Recursive Feature Elimination) | Recursively removes weakest features using RandomForest |

### 📌 Embedded Methods
Feature selection **built into the model** during training.

| Method | Description |
|---|---|
| Feature Importance | Uses RandomForest's built-in `feature_importances_` |
| L1 Regularization | LinearSVC with Lasso penalty zeros out weak features |

---

## 🏆 Results Summary

| Method | Accuracy | F1 Score | Features Used |
|---|---|---|---|
| All Features (Baseline) | 0.8313 | 0.8393 | 11 |
| Strong Corr Features | 0.8250 | 0.8343 | 7 |
| Subset Corr Features | 0.8188 | 0.8333 | 6 |
| **F-test (top 8)** | **0.8438** | **0.8521** | **8** ✅ |
| RFE (top 8) | 0.8281 | 0.8406 | 8 |
| Feature Importance | 0.8219 | 0.8328 | 6 |
| L1 Regularization | 0.8188 | 0.8314 | 10 |

### 🔑 Key Findings
- **Best Performance:** ANOVA F-test with top 8 features achieved the highest F1 Score of **0.852**, outperforming the baseline that used all 11 features
- **Best Efficiency:** Subset Correlation and Feature Importance methods achieved good performance with only **6 features**
- **Most Important Features** (consistently selected across all methods): `volatile acidity`, `alcohol`, `sulphates`, `chlorides`, `total sulfur dioxide`

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- VS Code with Python extension installed

### Steps

**1. Clone or download the project**

**2. Create and activate a virtual environment**
```bash
# Create
python3 -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the script**
```bash
python wine_feature_selection.py
```

---

## 📈 Output

Running the script will generate the following plots:
1. Distribution of Wine Quality Scores
2. Full Correlation Matrix Heatmap
3. Correlation Heatmap of Selected Features
4. ANOVA F-Score Bar Chart
5. Random Forest Feature Importance Plot
6. F1 Score Comparison Across All Methods
7. Feature Count vs F1 Score (Efficiency Trade-off)

---

## 🧠 Concepts Covered

- Binary classification on a real-world dataset
- Data preprocessing and feature engineering
- Filter, Wrapper, and Embedded feature selection methods
- Model evaluation using Accuracy, ROC, Precision, Recall, and F1 Score
- Visualizing correlations and feature importance

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Programming language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models & feature selection |
| matplotlib | Plotting |
| seaborn | Statistical visualization |

---

## 📚 References

- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [Scikit-learn Feature Selection Docs](https://scikit-learn.org/stable/modules/feature_selection.html)
- Cortez et al., 2009 – *Modeling wine preferences by data mining from physicochemical properties*