# PHE Fouling Dashboard

A Streamlit starter dashboard for a plate heat exchanger fouling project.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Deploy the app and upload your CSV from the dashboard sidebar.

## Important limitations
This starter app provides the five-section layout, dataset upload/preview, basic preprocessing, and baseline Random Forest/XGBoost classification when the target column is named `Fouling_Detected`.

Engineering calculations, validated fouling thresholds, regression-based Rf estimation, LSTM forecasting, SHAP explanations, severity scoring, and maintenance rules are not fully implemented yet. Confirm the actual CSV column names, units, target-label meaning, and clean-condition baseline before adding these features. Do not use the output as an operational maintenance decision without validation.
