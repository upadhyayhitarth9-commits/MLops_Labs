# =============================================================
# Feature Selection Lab: Wine Quality Dataset
# =============================================================
# Dataset: winequality-red.csv (local file)
# Target: quality (binary: 0 = low quality <=5, 1 = high quality >=6)
# Methods: Filter, Wrapper, and Embedded Feature Selection
# =============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# ── 2. LOAD DATASET ──────────────────────────────────────────
# Make sure winequality-red.csv is inside a 'data/' folder
# in the same directory as this script.

df = pd.read_csv('./data/winequality-red.csv', sep=',')

print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isna().sum())
print("\nDataset Preview:")
print(df.head())
print("\nStatistical Summary:")
print(df.describe())

# ── 3. EXPLORE THE TARGET VARIABLE ───────────────────────────
plt.figure(figsize=(7, 4))
sns.countplot(x='quality', data=df, palette='Blues_d')
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Binarize target:
#   0 = low quality  (score <= 5)
#   1 = high quality (score >= 6)
df['quality_label'] = (df['quality'] >= 6).astype(int)
df.drop('quality', axis=1, inplace=True)

print("\nClass distribution after binarizing:")
print(df['quality_label'].value_counts())

# ── 4. SPLIT FEATURES AND TARGET ─────────────────────────────
X = df.drop('quality_label', axis=1)
Y = df['quality_label']

print("\nFeatures:", list(X.columns))
print("Number of features:", X.shape[1])

# ── 5. HELPER FUNCTIONS ──────────────────────────────────────

def fit_model(X_train, Y_train):
    """Train a RandomForestClassifier."""
    model = RandomForestClassifier(criterion='entropy', random_state=47, n_estimators=100)
    model.fit(X_train, Y_train)
    return model

def calculate_metrics(model, X_test_scaled, Y_test):
    """Return evaluation metrics on the test set."""
    y_pred = model.predict(X_test_scaled)
    acc  = accuracy_score(Y_test, y_pred)
    roc  = roc_auc_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec  = recall_score(Y_test, y_pred)
    f1   = f1_score(Y_test, y_pred)
    return acc, roc, prec, rec, f1

def train_and_get_metrics(X, Y):
    """Split, scale, train, and evaluate."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = fit_model(X_train_s, Y_train)
    return calculate_metrics(model, X_test_s, Y_test)

def evaluate_model_on_features(X, Y):
    """Train model and return metrics as a DataFrame row."""
    acc, roc, prec, rec, f1 = train_and_get_metrics(X, Y)
    return pd.DataFrame(
        [[acc, roc, prec, rec, f1, X.shape[1]]],
        columns=["Accuracy", "ROC", "Precision", "Recall", "F1 Score", "Feature Count"]
    )

# ── 6. BASELINE – ALL FEATURES ───────────────────────────────
print("\n" + "="*55)
print("BASELINE: Training with all features")
print("="*55)

all_features_df = evaluate_model_on_features(X, Y)
all_features_df.index = ['All Features']
results = all_features_df.copy()
print(results)

# ── 7. CORRELATION MATRIX ────────────────────────────────────
plt.figure(figsize=(14, 12))
cor = df.corr()
sns.heatmap(cor, annot=True, fmt=".2f", cmap='Blues', linewidths=0.5)
plt.title("Correlation Matrix – Wine Quality Features")
plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════
# SECTION A: FILTER METHODS
# ════════════════════════════════════════════════════════════

# ── A1. Correlation with Target ───────────────────────────────
print("\n" + "="*55)
print("SECTION A1: Correlation with Target (threshold > 0.1)")
print("="*55)

cor_target = abs(cor['quality_label'])
relevant_features = cor_target[cor_target > 0.1]
names = [idx for idx, val in relevant_features.items()]
names.remove('quality_label')

print("Selected features:", names)

strong_df = evaluate_model_on_features(df[names], Y)
strong_df.index = ['Strong Corr Features']
results = pd.concat([results, strong_df])
print(results)

# ── A2. Remove Highly Inter-Correlated Features ───────────────
print("\n" + "="*55)
print("SECTION A2: Removing Inter-Correlated Features")
print("="*55)

plt.figure(figsize=(12, 10))
new_corr = df[names].corr()
sns.heatmap(new_corr, annot=True, fmt=".2f", cmap='Blues', linewidths=0.5)
plt.title("Correlation Among Target-Relevant Features")
plt.tight_layout()
plt.show()

# 'free sulfur dioxide' and 'total sulfur dioxide' are highly
# correlated (~0.67). We keep 'free sulfur dioxide' and drop
# 'total sulfur dioxide'.
features_to_drop = ['total sulfur dioxide']
subset_names = [x for x in names if x not in features_to_drop]
print("Features after removing inter-correlated ones:", subset_names)

subset_df = evaluate_model_on_features(df[subset_names], Y)
subset_df.index = ['Subset Corr Features']
results = pd.concat([results, subset_df])
print(results)

# ── A3. Univariate Selection – ANOVA F-test ───────────────────
print("\n" + "="*55)
print("SECTION A3: Univariate Selection – ANOVA F-test (top 8)")
print("="*55)

def univariate_selection(k=8):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_train_s, Y_train)

    feature_idx   = selector.get_support()
    feature_names = X.columns[feature_idx]

    # Plot F-scores for all features
    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    scores.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(f"ANOVA F-Scores – Top {k} features selected")
    plt.ylabel("F-Score")
    plt.tight_layout()
    plt.show()

    print("Selected features by F-test:", list(feature_names))
    return feature_names

univariate_names = univariate_selection(k=8)
uni_df = evaluate_model_on_features(df[univariate_names], Y)
uni_df.index = ['F-test (top 8)']
results = pd.concat([results, uni_df])
print(results)

# ════════════════════════════════════════════════════════════
# SECTION B: WRAPPER METHODS
# ════════════════════════════════════════════════════════════

# ── B1. Recursive Feature Elimination (RFE) ───────────────────
print("\n" + "="*55)
print("SECTION B1: Recursive Feature Elimination (RFE, top 8)")
print("="*55)

def run_rfe(n_features=8):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    model = RandomForestClassifier(criterion='entropy', random_state=47)
    rfe   = RFE(model, n_features_to_select=n_features)
    rfe.fit(X_train_s, Y_train)

    feature_names = X.columns[rfe.get_support()]
    ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()

    print("RFE Feature Rankings (1 = selected):")
    print(ranking)
    print("\nSelected features:", list(feature_names))
    return feature_names

rfe_names = run_rfe(n_features=8)
rfe_df = evaluate_model_on_features(df[rfe_names], Y)
rfe_df.index = ['RFE (top 8)']
results = pd.concat([results, rfe_df])
print(results)

# ════════════════════════════════════════════════════════════
# SECTION C: EMBEDDED METHODS
# ════════════════════════════════════════════════════════════

# ── C1. Feature Importances (Tree-based) ─────────────────────
print("\n" + "="*55)
print("SECTION C1: Feature Importance – Random Forest")
print("="*55)

def feature_importance_selection(threshold=0.07):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=47)
    model.fit(X_train_s, Y_train)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=True).plot(
        kind='barh', figsize=(10, 6), color='teal', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title("Feature Importances – Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.show()

    selector      = SelectFromModel(model, prefit=True, threshold=threshold)
    feature_idx   = selector.get_support()
    feature_names = X.columns[feature_idx]
    print("Selected features (importance > threshold):", list(feature_names))
    return feature_names

feat_imp_names = feature_importance_selection(threshold=0.07)
feat_imp_df = evaluate_model_on_features(df[feat_imp_names], Y)
feat_imp_df.index = ['Feature Importance']
results = pd.concat([results, feat_imp_df])
print(results)

# ── C2. L1 Regularization (Lasso via LinearSVC) ──────────────
print("\n" + "="*55)
print("SECTION C2: L1 Regularization")
print("="*55)

def run_l1_regularization(C=0.1):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=123)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    selection = SelectFromModel(LinearSVC(C=C, penalty='l1', dual=False, max_iter=5000))
    selection.fit(X_train_s, Y_train)

    feature_names = X.columns[selection.get_support()]
    print(f"L1 Regularization (C={C}) selected features:", list(feature_names))
    return feature_names

l1_names = run_l1_regularization(C=0.1)
l1_df = evaluate_model_on_features(df[l1_names], Y)
l1_df.index = ['L1 Regularization']
results = pd.concat([results, l1_df])

# ════════════════════════════════════════════════════════════
# SECTION D: FINAL COMPARISON
# ════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("FINAL RESULTS – All Feature Selection Methods")
print("="*65)
print(results.to_string())

# ── Plot 1: F1 Score Comparison ───────────────────────────────
plt.figure(figsize=(12, 5))
results['F1 Score'].plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("F1 Score Comparison Across Feature Selection Methods")
plt.ylabel("F1 Score")
plt.xticks(rotation=30, ha='right')
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.show()

# ── Plot 2: Feature Count vs F1 Score ────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(results['Feature Count'], results['F1 Score'],
            s=120, color='teal', edgecolors='black', zorder=5)
for i, label in enumerate(results.index):
    plt.annotate(label,
                 (results['Feature Count'].iloc[i], results['F1 Score'].iloc[i]),
                 textcoords="offset points", xytext=(6, 5), fontsize=9)
plt.xlabel("Number of Features")
plt.ylabel("F1 Score")
plt.title("Feature Count vs. F1 Score (Efficiency Trade-off)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =============================================================
# CONCLUSION – Discussion Questions
# -------------------------------------------------------------
# 1. Which method achieves the best F1 Score?
# 2. Which method gives the best trade-off between
#    performance and number of features?
# 3. How does L1 Regularization compare to tree-based
#    Feature Importance for this dataset?
# 4. Were any features consistently selected across ALL methods?
# =============================================================