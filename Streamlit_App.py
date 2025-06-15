import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="German Credit Risk Analyzer", layout="centered")
st.title("📊 German Credit Risk Analyzer")

tab1, tab2, tab3 = st.tabs(["📚 Train Model", "🔍 Predict", "🔎 Clustering"])

# --- Tab 1: Supervised Learning (Logistic Regression) ---
with tab1:
    st.header("📚 Train Model - Logistic Regression")

    uploaded_file = st.file_uploader("Upload CSV data (german_credit_data.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
        df["Credit amount"] = pd.to_numeric(df["Credit amount"], errors='coerce')

        df["Risk_binary"] = df["Credit amount"].apply(lambda x: "good" if x >= 10000 else "bad")
        le = LabelEncoder()
        df["Risk_binary"] = le.fit_transform(df["Risk_binary"])

        X = df[["Age", "Job", "Credit amount", "Duration", "Housing"]]
        y = df["Risk_binary"]

        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        split_ratio = st.selectbox("Pilih rasio test set", [0.3, 0.2, 0.35, 0.25])
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=split_ratio, random_state=42)

    if st.button("Train Model"):
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)

        st.success("✅ Model berhasil dilatih.")
        st.write("Akurasi training:", logreg.score(X_train, y_train))
        st.write("Akurasi testing:", logreg.score(X_test, y_test))

        y_pred = logreg.predict(X_test)

        st.subheader("📊 Confusion Matrix")
        st.text(confusion_matrix(y_test, y_pred))

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("📈 ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

# --- Tab 2: Prediction ---
with tab2:
    st.header("🔍 Predict")

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

    if st.button("Submit"):
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

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        st.write(f"📌 Prediksi Risiko: **{'Bad' if prediction else 'Good'}**")
        st.write(f"📈 Probabilitas: `{proba:.2f}`")

        st.markdown("---")
        st.subheader("🧾 Ringkasan Data Nasabah")
        display_data = input_data.copy()
        display_data["Housing"] = housing
        st.table(display_data)

        st.subheader("📊 Visualisasi Data Input")
        input_data_numeric = input_data.select_dtypes(include=np.number)
        numeric_data = input_data_numeric.iloc[0]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = sns.barplot(x=numeric_data.index, y=numeric_data.values, palette="viridis", ax=ax)

        for bar in bars.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(numeric_data.values) * 0.02,
                f'{height:.0f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color='black'
            )

        ax.set_title("Data Input Nasabah", fontsize=14, fontweight='bold')
        ax.set_ylabel("Nilai")
        ax.set_xlabel("Fitur")
        plt.xticks(rotation=30)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        st.pyplot(fig)

# --- Tab 3: Unsupervised Learning (KMeans) ---
with tab3:
    st.header("🔎 Clustering - KMeans")

    uploaded_file_km = st.file_uploader("Upload CSV data untuk Clustering", type=["csv"], key="clustering")
    if uploaded_file_km:
        df_km = pd.read_csv(uploaded_file_km)
        df_km.columns = df_km.columns.str.strip()
        df_km.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

        X_km = df_km[["Age", "Job", "Credit amount", "Duration"]].copy()
        X_km = X_km.dropna()
        X_km_scaled = StandardScaler().fit_transform(X_km)

        cluster_count = st.selectbox("Pilih jumlah cluster", [3, 4, 5, 6])
        kmeans = KMeans(n_clusters=cluster_count, random_state=42)
        df_km["Cluster"] = kmeans.fit_predict(X_km_scaled)

        st.subheader("Hasil Clustering")
        st.write(df_km[["Age", "Job", "Credit amount", "Duration", "Cluster"]].head())

        st.subheader("Visualisasi Clustering (PCA 2D)")
        pcs = PCA(n_components=2).fit_transform(X_km_scaled)
        df_km["PC1"] = pcs[:, 0]
        df_km["PC2"] = pcs[:, 1]

        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_km, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax2)
        ax2.set_title("Visualisasi Clustering (PCA)")
        st.pyplot(fig2)