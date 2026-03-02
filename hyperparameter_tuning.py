import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
y = le.fit_transform(raw['Disease'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')
print("\nTop 5 Configurations:")
print(results_df[['params', 'mean_test_score', 'std_test_score']].head())

