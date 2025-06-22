# --- Imports ---
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# --- Fetch dataset ---
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X = cdc_diabetes_health_indicators.data.features
Y = cdc_diabetes_health_indicators.data.targets
y = Y.values.ravel()

# --- Feature Types ---
binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
ordinal_features = ['GenHlth', 'Age', 'Education', 'Income']
ratio_features = ['BMI', 'MentHlth', 'PhysHlth']

# --- Preprocessing Pipeline ---
ordinal_encoder = OrdinalEncoder()
scaler = StandardScaler()
preprocessor = ColumnTransformer([
    ('ordinal', ordinal_encoder, ordinal_features),
    ('ratio', scaler, ratio_features)
], remainder='passthrough')

# --- Model Definition ---
model = RandomForestClassifier()

# --- Hyperparameter Grid ---
grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [None, 10]
}

# --- Build Model Pipeline ---
def build_model(preprocessor, model):
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])

# --- Sampling and Splitting Config ---
sampler = SMOTE(random_state=42)
X_sampled, y_sampled = sampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42)

# --- Evaluation ---
pipe = build_model(preprocessor, model)
grid_search = GridSearchCV(pipe, grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Best parameters: {best_params}")

y_test_pred = best_model.predict(X_test)

# --- Results ---
results = {
    "F1": f1_score(y_test, y_test_pred),
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "Best Params": best_params
}

print("\n--- Evaluation Results ---")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("\n--- Confusion Matrix ---")
print(conf_matrix)
