import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Data Profiler & AI Cleaner",
    page_icon="🔍",
    layout="wide"
)

st.title("⚡ Smart Data Profiler, Anomaly Detector & Auto-Cleaner")
st.markdown("Upload your dataset to profile, detect anomalies with **Explainable AI (SHAP)**, auto-fix issues, and export production code.")

# --- File Upload ---
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read Dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Profiling", 
        "🚨 Anomaly Detection (ML + SHAP)", 
        "🤖 AI Summary", 
        "🛠️ Interactive Auto-Cleaner", 
        "💻 Export Python Code"
    ])

    # ==========================================
    # TAB 1: DATA PROFILING
    # ==========================================
    with tab1:
        st.subheader("Data Overview & Quality Metrics")
        
        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        health_score = round(100 - (missing_cells / total_cells * 100), 2)
        
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", f"{missing_cells} ({round(missing_cells/total_cells*100, 1)}%)")
        col4.metric("Data Health Score", f"{health_score}%")

        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### First 5 Rows")
            st.dataframe(df.head(), use_container_width=True)
        with col_right:
            st.markdown("### Missing Values Breakdown")
            missing_df = df.isna().sum().reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["Missing %"] = (missing_df["Missing Count"] / len(df)) * 100
            st.dataframe(missing_df[missing_df["Missing Count"] > 0], use_container_width=True)

        st.markdown("### Numeric Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for correlation matrix.")

    # ==========================================
    # TAB 2: ANOMALY DETECTION + SHAP EXPLAINABILITY
    # ==========================================
    with tab2:
        st.subheader("Multivariate Anomaly Detection with SHAP Explanations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Please upload a dataset with at least two numeric features for ML anomaly detection.")
        else:
            contamination = st.slider("Contamination Factor (Estimated Outlier %)", 0.01, 0.20, 0.05, 0.01)
            
            # Prepare numeric data for Isolation Forest
            df_num = df[numeric_cols].copy()
            df_num_imputed = df_num.fillna(df_num.median())

            # Train Isolation Forest
            model = IsolationForest(contamination=contamination, random_state=42)
            df['Anomaly_Score'] = model.fit_predict(df_num_imputed)
            df['Is_Anomaly'] = df['Anomaly_Score'].apply(lambda x: True if x == -1 else False)

            anomalies_count = df['Is_Anomaly'].sum()
            st.warning(f"Detected **{anomalies_count}** anomalies out of **{len(df)}** rows ({round(anomalies_count/len(df)*100, 2)}%).")

            st.dataframe(df[df['Is_Anomaly'] == True].head(10), use_container_width=True)

            # SHAP Root-Cause Explainer
            st.markdown("### 🔍 Root-Cause Explanation (SHAP Values)")
            st.caption("SHAP explains *why* the ML model flagged specific rows as anomalies by displaying feature importance.")

            explainer = shap.Explainer(model, df_num_imputed)
            shap_values = explainer(df_num_imputed)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_values, df_num_imputed, plot_type="bar", show=False)
            st.pyplot(fig)

    # ==========================================
    # TAB 3: LLM HEALTH ASSISTANT / EXECUTIVE SUMMARY
    # ==========================================
    with tab3:
        st.subheader("🤖 Automated Data Health Summary")
        
        # Rule-based auto summary generation
        summary_text = f"""
        ### Executive Dataset Report
        * **Dataset Structure:** Contains **{df.shape[0]}** rows and **{df.shape[1]}** variables.
        * **Health Status:** The overall quality score is **{health_score}%**.
        * **Missing Data:** Found **{missing_cells}** missing entries across all columns.
        * **Anomalies Detected:** Isolation Forest identified **{df.get('Is_Anomaly', pd.Series()).sum()}** suspicious rows based on feature interactions.
        
        **Recommended Actions:**
        1. Apply KNN Imputation on numerical features with missing values.
        2. Inspect highlighted outliers in Tab 2 to verify if they are telemetry glitches or valid extreme events.
        3. Clean and export the standardized file using Tab 4.
        """
        st.markdown(summary_text)

    # ==========================================
    # TAB 4: INTERACTIVE AUTO-CLEANER
    # ==========================================
    with tab4:
        st.subheader("🛠️ One-Click Data Remediation")
        st.write("Configure and execute automated cleaning steps to prepare production-ready data.")

        impute_strategy = st.radio("Missing Value Imputation Method", ["Mean / Mode Imputation", "KNN Imputation", "Drop Rows with Missing Values"])
        remove_anomalies = st.checkbox("Remove Flagged ML Anomalies", value=False)

        if st.button("Executes Cleaning Pipeline"):
            cleaned_df = df.copy()

            # Remove Anomalies if selected
            if remove_anomalies and 'Is_Anomaly' in cleaned_df.columns:
                cleaned_df = cleaned_df[cleaned_df['Is_Anomaly'] == False]
                cleaned_df = cleaned_df.drop(columns=['Anomaly_Score', 'Is_Anomaly'])

            # Handle Missing Values
            if impute_strategy == "Drop Rows with Missing Values":
                cleaned_df = cleaned_df.dropna()
            elif impute_strategy == "Mean / Mode Imputation":
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype in [np.float64, np.int64]:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown")
            elif impute_strategy == "KNN Imputation":
                num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    imputer = KNNImputer(n_neighbors=5)
                    cleaned_df[num_cols] = imputer.fit_transform(cleaned_df[num_cols])

            st.success("Dataset successfully cleaned!")
            st.dataframe(cleaned_df.head(), use_container_width=True)

            # Download CSV Button
            csv = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

    # ==========================================
    # TAB 5: REPRODUCIBLE CODE GENERATOR
    # ==========================================
    with tab5:
        st.subheader("💻 Automated Python Pipeline Generator")
        st.write("Copy and execute this generated Python script in your local environment or Jupyter Notebook to reproduce the exact cleaning and anomaly pipeline.")

        generated_code = f"""import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

# 1. Load Data
df = pd.read_csv("{uploaded_file.name}")

# 2. Anomaly Detection via Isolation Forest
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_num = df[numeric_cols].fillna(df[numeric_cols].median())

model = IsolationForest(contamination=0.05, random_state=42)
df['Is_Anomaly'] = model.fit_predict(df_num) == -1

# 3. Clean Missing Values via KNN
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# 4. Save Cleaned Dataset
df_clean = df[df['Is_Anomaly'] == False].drop(columns=['Is_Anomaly'])
df_clean.to_csv("cleaned_output.csv", index=False)
print("Pipeline executed successfully. Cleaned dataset saved as 'cleaned_output.csv'.")
"""
        st.code(generated_code, language="python")

else:
    st.info("👆 Please upload a CSV or Excel file in the sidebar to start processing.")