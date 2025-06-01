# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dan bersihkan data
df = pd.read_csv("german_credit_data.csv")
df.columns = df.columns.str.strip()
df["Credit amount"] = pd.to_numeric(df["Credit amount"], errors='coerce')

# Buat target binary
df["Risk_binary"] = df["Credit amount"].apply(lambda x: "good" if x <= 10000 else "bad")
le = LabelEncoder()
df["Risk_binary"] = le.fit_transform(df["Risk_binary"])  # good=0, bad=1

# Fitur
X = df[["Age", "Job", "Credit amount", "Duration", "Housing"]]
y = df["Risk_binary"]

# Encode kategorik
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split & model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Simpan model dan scaler
with open("model_logreg.pkl", "wb") as f:
    pickle.dump(logreg, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model dan scaler berhasil disimpan.")
