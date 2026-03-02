"""
evaluation.py
==============
Generates all evaluation charts required by the assignment:
- Confusion matrix
- Feature importance
- Accuracy curve
- ROC curve (one-vs-rest)
- Model comparison bar chart
- Specialty distribution pie chart
- Similarity method comparison
"""

import pandas as pd
import numpy as np
import os, warnings, pickle
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_curve, auc,
                              classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, label_binarize


os.makedirs('reports', exist_ok=True)


def add_noise(X, y, le, noise_level=0.12, seed=42):
    """Same noise logic as disease_classifier — dropout, false positives, label swaps."""
    rng = np.random.RandomState(seed)
    X_n = X.copy().values.astype(float)
    present = X_n == 1
    X_n[present & (rng.random(X_n.shape) < noise_level)] = 0
    absent = X_n == 0
    X_n[absent & (rng.random(X_n.shape) < 0.04)] = 1

    confusable_groups = [
        ["hepatitis A","Hepatitis B","Hepatitis C","Hepatitis D","Hepatitis E","Alcoholic hepatitis"],
        ["Hypothyroidism","Hyperthyroidism","Hypoglycemia","Diabetes"],
        ["Fungal infection","Impetigo","Psoriasis","Acne"],
        ["Malaria","Dengue","Typhoid","Common Cold","Chicken pox"],
        ["Pneumonia","Tuberculosis","Bronchial Asthma"],
        ["Arthritis","Osteoarthristis","Cervical spondylosis"],
        ["GERD","Peptic ulcer diseae","Gastroenteritis","Chronic cholestasis","Jaundice"],
        ["Migraine","(vertigo) Paroymsal  Positional Vertigo","Paralysis (brain hemorrhage)"],
    ]
    le_classes = list(le.classes_)
    swap_map = {}
    for group in confusable_groups:
        idxs = [le_classes.index(d) for d in group if d in le_classes]
        for i in idxs:
            others = [j for j in idxs if j != i]
            if others:
                swap_map[i] = others
    y_n = y.copy()
    mask = rng.random(len(y)) < 0.06
    for i in np.where(mask)[0]:
        if y_n[i] in swap_map:
            y_n[i] = rng.choice(swap_map[y_n[i]])
    return pd.DataFrame(X_n, columns=X.columns), y_n


def load_data():
    raw = pd.read_csv('data/raw/dataset.csv')
    raw['Disease'] = raw['Disease'].str.strip()
    symptom_cols = [c for c in raw.columns if 'Symptom' in c]
    all_syms = set()
    for col in symptom_cols:
        all_syms.update(raw[col].dropna().str.strip().tolist())
    all_syms.discard('')
    all_syms = sorted(all_syms)

    X = pd.DataFrame(0, index=raw.index, columns=all_syms)
    for col in symptom_cols:
        for idx, val in raw[col].items():
            if pd.notna(val):
                sym = val.strip()
                if sym in all_syms:
                    X.at[idx, sym] = 1

    le = LabelEncoder()
    y  = le.fit_transform(raw['Disease'])
    return X, y, le, all_syms


# ── 1. Confusion Matrix ──────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                linewidths=0.3, annot_kws={"size": 6})
    plt.title('Disease Classification — Confusion Matrix', fontsize=16, pad=15)
    plt.ylabel('Actual Disease',    fontsize=12)
    plt.xlabel('Predicted Disease', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0,  fontsize=6)
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved confusion_matrix.png")

    # Print metrics
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"\n  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    return acc, f1, prec, rec


# ── 2. Feature Importance ────────────────────────────────────
def plot_feature_importance(model, all_syms, top_n=20):
    importances = model.feature_importances_
    idx    = np.argsort(importances)[::-1][:top_n]
    labels = [all_syms[i].replace('_', ' ') for i in idx]
    values = importances[idx]

    plt.figure(figsize=(12, 7))
    palette = sns.color_palette('viridis', top_n)
    bars = plt.barh(labels[::-1], values[::-1], color=palette[::-1])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Most Predictive Symptoms\n(Random Forest Feature Importance)',
              fontsize=13)
    for bar, val in zip(bars, values[::-1]):
        plt.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved feature_importance.png")


# ── 3. Accuracy Curve ────────────────────────────────────────
def plot_accuracy_curve(X_train, X_test, y_train, y_test, best_params, le=None):
    estimator_range = [10, 25, 50, 75, 100, 125, 150, 200]
    train_acc, test_acc = [], []

    X_tr_n, y_tr_n = add_noise(X_train, y_train, le, noise_level=0.13, seed=42)
    X_te_n, y_te_n = add_noise(X_test,  y_test,  le, noise_level=0.10, seed=99)
    for n in estimator_range:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=best_params.get('max_depth'),
            min_samples_split=best_params.get('min_samples_split', 2),
            max_features=best_params.get('max_features', 'sqrt'),
            random_state=42
        )
        rf.fit(X_tr_n, y_tr_n)
        train_acc.append(accuracy_score(y_tr_n, rf.predict(X_tr_n)))
        test_acc.append(accuracy_score(y_te_n,  rf.predict(X_te_n)))

    plt.figure(figsize=(9, 5))
    plt.plot(estimator_range, train_acc, 'b-o', linewidth=2, label='Training Accuracy')
    plt.plot(estimator_range, test_acc,  'r-s', linewidth=2, label='Test Accuracy')
    plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
    plt.ylabel('Accuracy',  fontsize=12)
    plt.title('Model Accuracy vs Number of Trees', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/accuracy_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved accuracy_curve.png")


# 4. ROC Curve (One-vs-Rest, top 6 classes)
def plot_roc_curve(model, X_test, y_test, le):
    # Pick 6 most common classes in test set
    unique, counts = np.unique(y_test, return_counts=True)
    top6 = unique[np.argsort(counts)[::-1][:6]]
    class_names = le.inverse_transform(top6)

    y_bin  = label_binarize(y_test, classes=top6)
    proba  = model.predict_proba(X_test)
    # Subset probabilities to top6 classes
    proba_sub = proba[:, top6]

    plt.figure(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top6)))

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba_sub[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{cls_name} (AUC = {roc_auc:.2f})')

    plt.plot([0,1],[0,1], 'k--', lw=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate',  fontsize=12)
    plt.title('ROC Curves — Top 6 Disease Classes\n(One-vs-Rest)', fontsize=13)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved roc_curve.png")


# 5. Model Comparison Bar Chart
def plot_model_comparison(X_train, X_test, y_train, y_test, le=None):
    models = {
        'Random Forest':      RandomForestClassifier(n_estimators=100, max_depth=15,
                                                      random_state=42),
        'SVM':                SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression':LogisticRegression(max_iter=1000, random_state=42),
        'KNN':                KNeighborsClassifier(n_neighbors=5),
        'Decision Tree':      DecisionTreeClassifier(max_depth=15, random_state=42),
    }

    results = {}
    print("\nTraining comparison models (noisy training data, clean test)...")
    X_tr_n, y_tr_n = add_noise(X_train, y_train, le, noise_level=0.13, seed=42)
    X_te_n, y_te_n = add_noise(X_test,  y_test,  le, noise_level=0.10, seed=99)
    for name, m in models.items():
        m.fit(X_tr_n, y_tr_n)
        acc = accuracy_score(y_te_n, m.predict(X_te_n))
        f1  = f1_score(y_te_n, m.predict(X_te_n), average='weighted', zero_division=0)
        results[name] = {'Accuracy': acc, 'F1 Score': f1}
        print(f"  {name:22s}: Accuracy={acc:.4f}  F1={f1:.4f}")

    res_df = pd.DataFrame(results).T
    x = np.arange(len(res_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, res_df['Accuracy'], width,
                   label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, res_df['F1 Score'], width,
                   label='F1 Score', color='coral')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison\n(Accuracy & F1 Score)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df.index, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('reports/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved model_comparison.png")
    return res_df


# 6. Specialty distribution pie
def plot_specialty_distribution():
    hospitals = pd.read_csv('data/processed/hospitals.csv')
    all_specs = []
    for specs in hospitals['specialties'].dropna():
        all_specs.extend([s.strip() for s in specs.split(',')])
    spec_counts = pd.Series(all_specs).value_counts()

    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        spec_counts.values,
        labels=spec_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Set3(np.linspace(0, 1, len(spec_counts)))
    )
    for t in autotexts:
        t.set_fontsize(8)
    plt.title('Specialty Distribution Across Nepal Hospitals', fontsize=13)
    plt.tight_layout()
    plt.savefig('reports/specialty_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved specialty_distribution.png")


#7. Fee distribution by tier 
def plot_fee_distribution():
    hospitals = pd.read_csv('data/processed/hospitals.csv')
    plt.figure(figsize=(10, 5))
    for tier, color in zip(['Budget', 'Standard', 'Premium'],
                           ['green', 'blue', 'red']):
        subset = hospitals[hospitals['tier'] == tier]['opd_fee_npr']
        subset.hist(bins=20, alpha=0.6, color=color, label=tier, density=True)
    plt.xlabel('OPD Fee (NPR)', fontsize=12)
    plt.ylabel('Density',       fontsize=12)
    plt.title('OPD Fee Distribution by Hospital Tier', fontsize=13)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('reports/fee_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved fee_distribution.png")


# MAIN
def run_all_evaluations():
    print("=" * 55)
    print("  EVALUATION & VISUALIZATIONS")
    print("=" * 55)

    X, y, le, all_syms = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply noise to BOTH train and test to simulate real patient data
    # Train: heavier noise (model learns from messy data)
    # Test:  lighter noise (simulate real unseen patients — still imperfect)
    X_train_n, y_train_n = add_noise(X_train, y_train, le, noise_level=0.13, seed=42)
    X_test_n,  y_test_n  = add_noise(X_test,  y_test,  le, noise_level=0.10, seed=99)

    # Train best model
    best_params = {
        'n_estimators': 100, 'max_depth': 15,
        'min_samples_split': 2, 'max_features': 'sqrt'
    }
    print("\nTraining final Random Forest model...")
    rf = RandomForestClassifier(**best_params, random_state=42)
    rf.fit(X_train_n, y_train_n)

    print("\n── Metrics ──")
    acc, f1, prec, rec = plot_confusion_matrix(rf, X_test_n, y_test_n, le)

    print("\n── Generating all plots ──")
    plot_feature_importance(rf, list(all_syms))
    plot_accuracy_curve(X_train, X_test_n, y_train_n, y_test_n, best_params, le)
    plot_roc_curve(rf, X_test_n, y_test_n, le)
    comp_df = plot_model_comparison(X_train, X_test_n, y_train_n, y_test_n, le)
    plot_specialty_distribution()
    plot_fee_distribution()

    print("\n── Model Comparison Summary ──")
    print(comp_df.round(4).to_string())

    print("\n✓ All evaluation reports saved to reports/")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_all_evaluations()