# Smart Data Profiler

A Streamlit app for exploring CSV and Excel datasets, reviewing data quality, detecting potential anomalies, and exporting a cleaned CSV or reproducible Python pipeline.

## Run locally

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If Windows blocks a native package such as NumPy under Smart App Control, keep that protection enabled and use the hosted deployment steps below instead.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and `utils/__init__.py`.
2. Do not upload `.venv/`, private datasets, or generated cleaned files. `.gitignore` excludes these by default.
3. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
4. Select the repository, its branch, and `app.py`, then deploy.
5. Open the app URL and test it with a non-sensitive CSV or Excel file.

Uploaded datasets are processed by the hosted app. Do not upload sensitive data unless you have reviewed the hosting provider's privacy and retention terms.

## Included features

- CSV and Excel upload
- Dataset overview, missing-value summary, and numeric correlation heatmap
- Isolation Forest anomaly detection with SHAP feature explanations
- Missing-value imputation and CSV download
- Generated Python example for reproducing a cleaning pipeline
