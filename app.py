import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from xgboost import XGBClassifier

st.set_page_config(
    page_title="PHE Fouling Dashboard",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Plate Heat Exchanger Fouling Dashboard")
st.caption(
    "Engineering Performance | Machine Learning | "
    "SHAP | Forecasting | Maintenance"
)

st.sidebar.header("Dashboard Settings")
file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

if file is None:
    st.info("Upload your heat exchanger CSV dataset to begin.")
    st.stop()

df = pd.read_csv(file)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Data Input",
    "2. Preprocessing",
    "3. Engineering Performance",
    "4. Fouling Prediction & SHAP",
    "5. Maintenance Support"
])

# 1. DATA INPUT
with tab1:
    st.header("1. Data Input")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", f"{len(df):,}")
    c2.metric("Dataset columns", len(df.columns))
    c3.metric("Missing cells", f"{df.isna().sum().sum():,}")

    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

    row_num = st.number_input(
        "Select dataset row (starts at 0)",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    st.subheader("Selected record")
    st.dataframe(df.iloc[[int(row_num)]], use_container_width=True)

# 2. PREPROCESSING
with tab2:
    st.header("2. Data Preprocessing")

    missing = df.isna().sum()
    st.subheader("Missing values by column")
    st.dataframe(missing.rename("Missing count"))

    st.write("Original shape:", df.shape)
    cleaned = df.dropna().drop_duplicates()
    st.write("Cleaned shape:", cleaned.shape)
    st.write("Rows removed:", len(df) - len(cleaned))

    st.subheader("Cleaned dataset preview")
    st.dataframe(cleaned.head(100), use_container_width=True)

    st.caption(
        "Review missingness before deleting rows. Consider suitable imputation "
        "when dropping rows would remove too much useful data. Noise reduction "
        "must preserve genuine operating conditions."
    )

# 3. ENGINEERING PERFORMANCE
with tab3:
    st.header("3. Engineering Performance")
    st.write(
        "Inspect available numerical fields and verify their units before "
        "calculating engineering parameters."
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Available numerical columns:", numeric_cols)

    if numeric_cols:
        selected_col = st.selectbox(
            "Inspect an engineering parameter", numeric_cols
        )
        value = df.iloc[int(row_num)][selected_col]
        st.metric("Selected row value", str(value))

    st.info(
        "Next implementation: heat duty, LMTD, overall heat-transfer "
        "coefficient U, pressure drop and fouling resistance Rf using "
        "verified units, effective heat-transfer area and a validated clean baseline."
    )

# 4. PREDICTION AND EXPLAINABILITY
with tab4:
    st.header("4. Fouling Prediction & Explainability")
    target = "Fouling_Detected"

    if target not in df.columns:
        st.warning(
            f"Target column '{target}' was not found. Check the actual CSV column names."
        )
    else:
        data = df.copy()
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target])
        data = data[data[target].isin([0, 1])]

        X = data.select_dtypes(include=np.number).drop(
            columns=[target], errors="ignore"
        )
        y = data[target].astype(int)

        if len(X.columns) == 0 or y.nunique() < 2:
            st.warning(
                "At least one numeric feature and both target classes are required."
            )
        else:
            X = X.replace([np.inf, -np.inf], np.nan)
            imputer = SimpleImputer(strategy="median")
            X = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

            if y.value_counts().min() < 2:
                st.warning(
                    "Each class needs at least two records for a stratified train/test split."
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                models = {
                    "Random Forest": RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    ),
                    "XGBoost": XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=42
                    )
                }

                results = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    results[name] = {
                        "model": model,
                        "accuracy": accuracy_score(y_test, pred),
                        "precision": precision_score(
                            y_test, pred, zero_division=0
                        ),
                        "recall": recall_score(
                            y_test, pred, zero_division=0
                        ),
                        "f1": f1_score(y_test, pred, zero_division=0)
                    }

                st.subheader("Model comparison")
                st.dataframe(pd.DataFrame([
                    {
                        "Model": name,
                        "Accuracy": r["accuracy"],
                        "Precision": r["precision"],
                        "Recall": r["recall"],
                        "F1-score": r["f1"]
                    }
                    for name, r in results.items()
                ]), use_container_width=True)

                model_name = st.selectbox("Select model", list(results.keys()))
                model = results[model_name]["model"]

                st.subheader("Selected row prediction")
                original_row = df.iloc[[int(row_num)]]
                features = original_row.reindex(columns=X.columns)
                features = features.replace([np.inf, -np.inf], np.nan)
                features = pd.DataFrame(
                    imputer.transform(features), columns=X.columns
                )

                prediction = int(model.predict(features)[0])
                st.metric(
                    "Prediction",
                    "Fouled (1)" if prediction == 1 else "Clean (0)"
                )

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    st.metric(
                        "Probability of fouling",
                        f"{probabilities[1] * 100:.1f}%"
                    )

                st.subheader("Feature importance")
                importance = pd.Series(
                    model.feature_importances_, index=X.columns
                ).sort_values(ascending=False)
                st.bar_chart(importance.head(10))

                st.subheader("Confusion matrix")
                st.write(confusion_matrix(y_test, model.predict(X_test)))

                st.caption(
                    "Feature importance is not SHAP. Add SHAP separately for "
                    "local and global model explanations."
                )

# 5. MAINTENANCE SUPPORT
with tab5:
    st.header("5. Maintenance Support")
    st.write(
        "Recommendations should be generated from validated engineering "
        "thresholds, predicted fouling status, trends and model explanations."
    )
    st.info(
        "Connect the maintenance rules after verifying the dataset columns, "
        "engineering thresholds and physical assumptions."
    )

st.divider()
st.caption(
    "Academic prototype. Validate labels, units, clean baseline and model "
    "performance before operational use."
)
