"""Quick Random Forest training script."""
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("🌲 Random Forest - Chargement des données...")

# Load data
X_train = pd.read_csv("data/clean/decision_tree_model/train_features.csv", index_col=0)
X_test = pd.read_csv("data/clean/decision_tree_model/test_features.csv", index_col=0)
y_train = pd.read_csv("data/clean/decision_tree_model/train_labels.csv", index_col=0).squeeze()
y_test = pd.read_csv("data/clean/decision_tree_model/test_labels.csv", index_col=0).squeeze()

print(f"✓ Données chargées: Train {X_train.shape}, Test {X_test.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"✓ {len(label_encoder.classes_)} classes")

# Train Random Forest (optimisé pour vitesse)
print("\n🚀 Entraînement Random Forest...")
print("   - 50 arbres (rapide)")
print("   - max_depth: 20")
print("   - Parallélisé")

model = RandomForestClassifier(
    n_estimators=50,  # Seulement 50 arbres pour la vitesse
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=0.7,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

import time
start = time.time()
model.fit(X_train.values, y_train_encoded)
duration = time.time() - start

print(f"✓ Entraînement terminé en {duration:.1f}s")

# Evaluate
print("\n📊 Évaluation...")
y_pred = model.predict(X_test.values)
train_pred = model.predict(X_train.values)

accuracy = accuracy_score(y_test_encoded, y_pred)
train_accuracy = accuracy_score(y_train_encoded, train_pred)
f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test_encoded, y_pred, average='weighted')

print(f"\n✨ RÉSULTATS RANDOM FOREST:")
print(f"   Test Accuracy:  {accuracy:.1%}")
print(f"   Train Accuracy: {train_accuracy:.1%}")
print(f"   Overfitting:    {(train_accuracy - accuracy):.1%}")
print(f"   F1-weighted:    {f1_weighted:.1%}")
print(f"   Precision:      {precision:.1%}")
print(f"   Recall:         {recall:.1%}")

# Save
Path("models/RandomForest/artefacts").mkdir(parents=True, exist_ok=True)
Path("models/RandomForest/metrics").mkdir(parents=True, exist_ok=True)

with open("models/RandomForest/artefacts/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/RandomForest/artefacts/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

metrics = {
    "accuracy": float(accuracy),
    "train_accuracy": float(train_accuracy),
    "overfitting_gap": float(train_accuracy - accuracy),
    "f1_weighted": float(f1_weighted),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "n_estimators": 50,
    "max_depth": 20
}

with open("models/RandomForest/metrics/metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Modèle sauvegardé dans models/RandomForest/")
