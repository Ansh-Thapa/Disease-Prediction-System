"""
disease_classifier.py
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# DISEASE → SPECIALTY MAP  (same as data_preparation.py)
DISEASE_SPECIALTY_MAP = {
    'Fungal infection':                        'Dermatology',
    'Allergy':                                 'General Medicine',
    'GERD':                                    'Gastroenterology',
    'Chronic cholestasis':                     'Gastroenterology',
    'Drug Reaction':                           'General Medicine',
    'Peptic ulcer diseae':                     'Gastroenterology',
    'AIDS':                                    'Infectious Disease',
    'Diabetes':                                'Endocrinology',
    'Gastroenteritis':                         'Gastroenterology',
    'Bronchial Asthma':                        'Pulmonology',
    'Hypertension':                            'Cardiology',
    'Migraine':                                'Neurology',
    'Cervical spondylosis':                    'Orthopedics',
    'Paralysis (brain hemorrhage)':            'Neurology',
    'Jaundice':                                'Gastroenterology',
    'Malaria':                                 'Infectious Disease',
    'Chicken pox':                             'Infectious Disease',
    'Dengue':                                  'Infectious Disease',
    'Typhoid':                                 'Infectious Disease',
    'hepatitis A':                             'Gastroenterology',
    'Hepatitis B':                             'Gastroenterology',
    'Hepatitis C':                             'Gastroenterology',
    'Hepatitis D':                             'Gastroenterology',
    'Hepatitis E':                             'Gastroenterology',
    'Alcoholic hepatitis':                     'Gastroenterology',
    'Tuberculosis':                            'Pulmonology',
    'Common Cold':                             'General Medicine',
    'Pneumonia':                               'Pulmonology',
    'Dimorphic hemmorhoids(piles)':            'General Surgery',
    'Heart attack':                            'Cardiology',
    'Varicose veins':                          'General Surgery',
    'Hypothyroidism':                          'Endocrinology',
    'Hyperthyroidism':                         'Endocrinology',
    'Hypoglycemia':                            'Endocrinology',
    'Osteoarthristis':                         'Orthopedics',
    'Arthritis':                               'Orthopedics',
    '(vertigo) Paroymsal  Positional Vertigo': 'Neurology',
    'Acne':                                    'Dermatology',
    'Urinary tract infection':                 'Urology',
    'Psoriasis':                               'Dermatology',
    'Impetigo':                                'Dermatology',
}


class DiseaseClassifier:
    def __init__(self):
        self.model         = None
        self.all_symptoms  = []
        self.label_encoder = LabelEncoder()
        self.X_test        = None
        self.y_test        = None
        self.y_pred        = None

    # ── Build feature matrix ────────────────────────────────
    def _build_features(self, disease_df):
        symptom_cols = [c for c in disease_df.columns if 'Symptom' in c]
        all_syms = set()
        for col in symptom_cols:
            all_syms.update(disease_df[col].dropna().str.strip().tolist())
        all_syms.discard('')
        self.all_symptoms = sorted(all_syms)
        X = pd.DataFrame(0, index=disease_df.index, columns=self.all_symptoms)
        for col in symptom_cols:
            for idx, val in disease_df[col].items():
                if pd.notna(val):
                    sym = val.strip()
                    if sym in self.all_symptoms:
                        X.at[idx, sym] = 1
        return X

    def _add_realistic_noise(self, X, y, noise_level=0.12, seed=42):
        """
        Simulate realistic clinical data messiness:
          1. Symptom dropout  - patients forget to mention symptoms (1->0)
          2. False positives  - patients misreport symptoms (0->1)
          3. Label noise      - confusable disease pairs swap labels (~6%)
        """
        rng = np.random.RandomState(seed)
        X_noisy = X.copy().values.astype(float)
        n_samples, n_features = X_noisy.shape

        # 1. Dropout:
        present = X_noisy == 1
        dropout = rng.random(X_noisy.shape) < noise_level
        X_noisy[present & dropout] = 0

        # 2. False positives:
        absent = X_noisy == 0
        false_pos = rng.random(X_noisy.shape) < 0.04
        X_noisy[absent & false_pos] = 1

        # 3. Label noise:
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
        le_classes = list(self.label_encoder.classes_)
        swap_map = {}
        for group in confusable_groups:
            idxs = [le_classes.index(d) for d in group if d in le_classes]
            for i in idxs:
                others = [j for j in idxs if j != i]
                if others:
                    swap_map[i] = others

        y_noisy = y.copy()
        label_noise_mask = rng.random(n_samples) < 0.06
        for i in np.where(label_noise_mask)[0]:
            orig = y_noisy[i]
            if orig in swap_map:
                y_noisy[i] = rng.choice(swap_map[orig])

        return pd.DataFrame(X_noisy, columns=self.all_symptoms), y_noisy

    # ── Train ───────────────────────────────────────────────
    def train(self):
        print("=" * 55)
        print("  STAGE 1: DISEASE CLASSIFIER TRAINING")
        print("=" * 55)

        raw = pd.read_csv('data/raw/dataset.csv')
        raw['Disease'] = raw['Disease'].str.strip()

        X = self._build_features(raw)
        y = self.label_encoder.fit_transform(raw['Disease'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        self.X_test = X_test
        self.y_test = y_test

        # Apply noise to both train and test to simulate real clinical data
        print("Applying realistic noise to training and test data...")
        X_train, y_train = self._add_realistic_noise(X_train, y_train, noise_level=0.13, seed=42)
        X_test,  y_test  = self._add_realistic_noise(X_test,  y_test,  noise_level=0.10, seed=99)
        self.X_test = X_test
        self.y_test = y_test
        print(f"Training samples : {len(X_train)} (with symptom dropout + label noise)")
        print(f"Test samples     : {len(X_test)}  (with lighter noise - unseen patients)")
        print(f"Unique diseases  : {len(self.label_encoder.classes_)}")
        print(f"Total symptoms   : {len(self.all_symptoms)}")

        # ── Hyperparameter tuning ───────────────────────────
        print("\nRunning GridSearchCV (5-fold CV)...")
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth':    [10, 15, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2'],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, cv=cv,
            scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)

        print(f"Best params      : {grid_search.best_params_}")
        print(f"Best CV accuracy : {grid_search.best_score_:.4f} "
              f"({grid_search.best_score_*100:.2f}%)")

        self.model = grid_search.best_estimator_

        # ── Evaluation ──────────────────────────────────────
        self.y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, self.y_pred)
        f1  = f1_score(y_test, self.y_pred, average='weighted')

        print(f"\nTest Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
        print(f"Weighted F1      : {f1:.4f}")

        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"CV Scores        : {cv_scores}")
        print(f"Mean CV Score    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        print("\nClassification Report (top diseases):")
        report = classification_report(
            y_test, self.y_pred,
            target_names=self.label_encoder.classes_, zero_division=0)
        print(report)

        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/disease_classifier.pkl', 'wb') as f:
            pickle.dump({
                'model':        self.model,
                'symptoms':     self.all_symptoms,
                'encoder':      self.label_encoder,
            }, f)
        print("Model saved → models/disease_classifier.pkl")
        return self

    # ── Predict ─────────────────────────────────────────────
    def predict(self, user_symptoms: list):
        """
        user_symptoms : list of symptom strings
        Returns: { disease, specialty, confidence, all_probs }
        """
        vec = pd.DataFrame([[0] * len(self.all_symptoms)],
                           columns=self.all_symptoms)
        for sym in user_symptoms:
            sym = sym.strip().lower().replace(' ', '_')
            if sym in self.all_symptoms:
                vec[sym] = 1

        pred_enc  = self.model.predict(vec)[0]
        proba     = self.model.predict_proba(vec)[0]
        disease   = self.label_encoder.inverse_transform([pred_enc])[0]
        specialty = DISEASE_SPECIALTY_MAP.get(disease, 'General Medicine')
        confidence = proba.max()

        top5 = sorted(
            zip(self.label_encoder.classes_, proba),
            key=lambda x: x[1], reverse=True
        )[:5]

        return {
            'disease':    disease,
            'specialty':  specialty,
            'confidence': confidence,
            'top5':       top5,
        }

    # ── Plots ────────────────────────────────────────────────
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(18, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    linewidths=0.3)
        plt.title('Disease Classification — Confusion Matrix', fontsize=16, pad=15)
        plt.ylabel('Actual Disease', fontsize=12)
        plt.xlabel('Predicted Disease', fontsize=12)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0,  fontsize=7)
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved → reports/confusion_matrix.png")

    def plot_feature_importance(self):
        importances = self.model.feature_importances_
        top_n = 20
        idx   = np.argsort(importances)[::-1][:top_n]
        top_syms  = [self.all_symptoms[i] for i in idx]
        top_imps  = importances[idx]

        plt.figure(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        bars = plt.barh(top_syms[::-1], top_imps[::-1], color=colors[::-1])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Most Important Symptoms\n(Feature Importance)', fontsize=14)
        plt.tight_layout()
        plt.savefig('reports/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved → reports/feature_importance.png")

    def plot_accuracy_curve(self):
        """Accuracy vs n_estimators curve using same noise as evaluation."""
        estimator_range = [10, 25, 50, 75, 100, 125, 150, 200]
        raw = pd.read_csv('data/raw/dataset.csv')
        raw['Disease'] = raw['Disease'].str.strip()
        X = self._build_features(raw)
        y = self.label_encoder.transform(raw['Disease'])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply same noise as training pipeline
        X_tr_n, y_tr_n = self._add_realistic_noise(X_tr, y_tr, noise_level=0.13, seed=42)
        X_te_n, y_te_n = self._add_realistic_noise(X_te, y_te, noise_level=0.10, seed=99)

        train_acc, test_acc = [], []
        best_p = self.model.get_params()
        for n in estimator_range:
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=best_p.get('max_depth'),
                min_samples_split=best_p.get('min_samples_split', 2),
                max_features=best_p.get('max_features', 'sqrt'),
                random_state=42
            )
            rf.fit(X_tr_n, y_tr_n)
            train_acc.append(accuracy_score(y_tr_n, rf.predict(X_tr_n)))
            test_acc.append(accuracy_score(y_te_n,  rf.predict(X_te_n)))

        plt.figure(figsize=(9, 5))
        plt.plot(estimator_range, train_acc, 'b-o', label='Training Accuracy')
        plt.plot(estimator_range, test_acc,  'r-s', label='Test Accuracy')
        plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracy vs Number of Trees', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/accuracy_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved → reports/accuracy_curve.png")

    @classmethod
    def load(cls):
        with open('models/disease_classifier.pkl', 'rb') as f:
            data = pickle.load(f)
        obj = cls()
        obj.model        = data['model']
        obj.all_symptoms = data['symptoms']
        obj.label_encoder = data['encoder']
        return obj


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    clf = DiseaseClassifier()
    clf.train()
    clf.plot_confusion_matrix()
    clf.plot_feature_importance()
    clf.plot_accuracy_curve()
    print("\n✓ Disease Classifier training complete!")