"""
setup_and_train.py
===================
Run this ONCE to:
1. Prepare / enrich hospital data
2. Train disease classifier
3. Generate all evaluation plots
"""

import os
import sys

# Always run from project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src')
print("NEPAL HOSPITAL FINDER — SETUP & TRAINING")

# Step 1 — Data preparation
print("STEP 1/3 — Data Preparation")
print("─" * 50)
from src.data_preparation import prepare_hospitals, prepare_disease_data
prepare_hospitals()
prepare_disease_data()

# Step 2 — Train disease classifier
print("\nSTEP 2/3 — Train Disease Classifier")
print("─" * 50)
from src.disease_classifier import DiseaseClassifier
clf = DiseaseClassifier()
clf.train()
clf.plot_confusion_matrix()
clf.plot_feature_importance()
clf.plot_accuracy_curve()

# Step 3 — Evaluation & visualizations
print("\nSTEP 3/3 — Evaluation & Visualizations")
print("─" * 50)
from src.evaluation import run_all_evaluations
run_all_evaluations()
print("✓ SETUP COMPLETE!")
print("Run 'python main.py' to start the Hospital Finder")

