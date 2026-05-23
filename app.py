import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from xgboost import XGBClassifier

# --- load data ---
@st.cache_data
def load_data():
    results = pd.read_csv('results.csv')
    X_test = pd.read_csv('X_test.csv')
    return results, X_test

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

results_df, X_test_df = load_data()
model, scaler = load_model()

# --- sidebar ---
st.sidebar.title("Fraud Detection System")
st.sidebar.markdown("By Devendra Patil")
page = st.sidebar.radio("Navigate", ["Overview", "Transaction Explorer", "SHAP Explainer"])

# =====================
# PAGE 1 — OVERVIEW
# =====================
if page == "Overview":
    st.title("Fraud Detection Dashboard")
    st.markdown("Real time fraud monitoring system built using XGBoost + SHAP")

    total = len(results_df)
    total_fraud = int(results_df['actual'].sum())
    detection_rate = round(results_df[results_df['actual']==1]['fraud_probability'].apply(lambda x: 1 if x >= 0.35 else 0).sum() / total_fraud * 100, 2)
    avg_fraud_amt = round(results_df[results_df['actual']==1]['TransactionAmt'].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Total Fraud Cases", f"{total_fraud:,}")
    col3.metric("Detection Rate", f"{detection_rate}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amt}")

    st.markdown("---")

    st.subheader("Risk Tier Distribution")
    tier_counts = results_df['risk_tier'].value_counts().reset_index()
    tier_counts.columns = ['Risk Tier', 'Count']
    st.dataframe(tier_counts, use_container_width=True)

    st.subheader("Fraud Count by Hour of Day")
    hour_fraud = results_df[results_df['actual']==1]['HourOfDay'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hour_fraud.index, hour_fraud.values, color='tomato')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Fraud Count')
    ax.set_title('Fraud by Hour of Day')
    st.pyplot(fig)

# =====================
# PAGE 2 — TRANSACTION EXPLORER
# =====================
elif page == "Transaction Explorer":
    st.title("Transaction Explorer")
    st.markdown("Search and filter transactions with live risk scores")

    col1, col2 = st.columns(2)
    with col1:
        tier_filter = st.multiselect("Filter by Risk Tier",
                                      options=['Clear', 'Suspicious', 'Critical Risk'],
                                      default=['Suspicious', 'Critical Risk'])
    with col2:
        min_prob = st.slider("Minimum Fraud Probability", 0.0, 1.0, 0.4)

    filtered = results_df[
        (results_df['risk_tier'].isin(tier_filter)) &
        (results_df['fraud_probability'] >= min_prob)
    ]

    st.markdown(f"Showing **{len(filtered):,}** transactions")

    display_cols = ['TransactionAmt', 'fraud_probability', 'risk_tier', 'actual', 'HourOfDay']
    st.dataframe(filtered[display_cols].sort_values('fraud_probability', ascending=False).head(500),
                 use_container_width=True)

# =====================
# PAGE 3 — SHAP EXPLAINER
# =====================
elif page == "SHAP Explainer":
    st.title("SHAP Transaction Explainer")
    st.markdown("Enter a row number to see why the model flagged it")

    row_num = st.number_input("Enter row number (0 to 118107)",
                               min_value=0, max_value=len(X_test_df)-1, value=0)

    if st.button("Explain Transaction"):
        with st.spinner("Computing SHAP values..."):
            row = X_test_df.iloc[row_num:row_num+1]
            prob = results_df.iloc[row_num]['fraud_probability']
            tier = results_df.iloc[row_num]['risk_tier']

            st.markdown(f"**Fraud Probability:** {prob:.4f}")
            st.markdown(f"**Risk Tier:** {tier}")

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(row)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=row.iloc[0],
                feature_names=X_test_df.columns.tolist()
            ), show=False)
            st.pyplot(fig)

            st.subheader("Plain English Explanation")
            if prob >= 0.75:
                st.error(f"This transaction is very likely FRAUD (probability: {prob:.2%}). The model is highly confident based on the transaction pattern.")
            elif prob >= 0.40:
                st.warning(f"This transaction is SUSPICIOUS (probability: {prob:.2%}). Recommend manual review.")
            else:
                st.success(f"This transaction appears LEGITIMATE (probability: {prob:.2%}). Low fraud risk.")