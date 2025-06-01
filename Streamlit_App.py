import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.title("🎯 German Credit Risk Classifier")

tab1, tab2 = st.tabs(["📚 Train Model", "🔍 Predict"])

with tab1:
    st.header("Train Model")

    uploaded_file = st.file_uploader("Upload CSV data (german_credit_data.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df["Credit amount"] = pd.to_numeric(df["Credit amount"], errors='coerce')

        # Binary target
        df["Risk_binary"] = df["Credit amount"].apply(lambda x: "good" if x >= 10000 else "bad")
        le = LabelEncoder()
        df["Risk_binary"] = le.fit_transform(df["Risk_binary"])

        X = df[["Age", "Job", "Credit amount", "Duration", "Housing"]]
        y = df["Risk_binary"]

        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)

        # Simpan model
        with open("model_logreg.pkl", "wb") as f:
            pickle.dump(logreg, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        st.success("✅ Model dan scaler berhasil disimpan.")
        st.write("Akurasi training:", logreg.score(X_train, y_train))
        st.write("Akurasi testing:", logreg.score(X_test, y_test))

with tab2:
    st.header("Predict")

    try:
        with open("model_logreg.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ Model belum dilatih. Silakan train model dulu di tab pertama.")
        st.stop()

    st.subheader("Masukkan Data Nasabah")
    age = st.slider("Age", 18, 75, 30)
    job = st.selectbox("Job", [0, 1, 2, 3])
    credit_amount = st.number_input("Credit Amount", value=2000)
    duration = st.number_input("Duration (months)", value=12)
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    housing_encoded = {"own": 2, "free": 0, "rent": 1}[housing]

    input_data = pd.DataFrame([{
        "Age": age,
        "Job": job,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Housing": housing_encoded
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][prediction]

    st.write(f"📊 Prediksi Risiko: **{'Bad' if prediction else 'Good'}**")
    st.write(f"Probabilitas: `{proba:.2f}`")
