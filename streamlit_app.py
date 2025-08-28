import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import clickhouse_connect

# Load saved model and scaler
@st.cache_resource
def load_model_scaler():
    model = joblib.load("svm_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_scaler()


# Function to preprocess the ClickHouse data to fit the model
def preprocess(df):
    # One-hot encoding transaction type
    transaction_type = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df.drop('type', axis=1), transaction_type], axis=1)

    # Feature engineering
    df['balance_diff'] = df.oldbalanceOrg - df.newbalanceOrig

    # Select features used in the model
    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_diff'] + \
               [col for col in df.columns if col.startswith('type_')]

    # Handle any missing columns due to one-hot encoding mismatch
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


client = clickhouse_connect.get_client(
    host=st.secrets["clickhouse"]["host"],
    user=st.secrets["clickhouse"]["user"],
    password=st.secrets["clickhouse"]["password"],
    secure=True,
    database='MySQL-CDC'  # Adjust if your DB name differs
)


# Assuming query_df is fetched externally, simulate this here for UI sake.
# Replace this stub with your actual query_df loading code.
# For example, you could place your query_df fetch function here.
def load_clickhouse_data():
    query = f"""
    select * from credit_card_transaction_data order by _peerdb_synced_at desc limit 100
    """

    result = client.query(query)
    df_fetched = pd.DataFrame(result.result_rows, columns=result.column_names)
    df_fetched = df_fetched.drop('isFlaggedFraud', axis=1)
    # This is a placeholder. Replace with actual db fetch.
    # e.g. query_df = fetch_data_from_clickhouse(...)
    return df_fetched.copy()

st.title("Credit Card Transaction Fraud Detection")

# Fetch and preprocess data
df = load_clickhouse_data()
if df.empty:
    st.warning("No transaction data available.")
    st.stop()

X = preprocess(df)

# Scale features and predict fraud probabilities and classes
X_scaled = scaler.transform(X)
fraud_proba = model.predict_proba(X_scaled)[:, 1]
fraud_pred = model.predict(X_scaled)

df['Fraud Probability'] = fraud_proba
df['Predicted Fraud'] = fraud_pred

# Display summary statistics
total_transactions = len(df)
total_fraud = fraud_pred.sum()
fraud_rate = total_fraud / total_transactions * 100

st.metric("Total Transactions", total_transactions)
st.metric("Predicted Fraudulent Transactions", total_fraud)
st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

# Interactive filter by prediction
filter_option = st.selectbox("Filter transactions by prediction:",  ["All", "Not Fraud", "Fraud"])

if filter_option == "Not Fraud":
    df_filtered = df[df['Predicted Fraud'] == 0]
elif filter_option == "Fraud":
    df_filtered = df[df['Predicted Fraud'] == 1]
else:
    df_filtered = df

# Show table with highlighting fraud rows
def highlight_fraud(row):
    return ['background-color: #f87171' if v == 1 else '' for v in row['Predicted Fraud':'Predicted Fraud']]

# We highlight rows entirely for simplicity here on predicted fraud
def highlight_row(row):
    if row['Predicted Fraud'] == 1:
        return ['background-color: #fddede'] * len(row)
    else:
        return [''] * len(row)

st.write(f"Showing {len(df_filtered)} transactions")

st.dataframe(df_filtered.style.apply(highlight_row, axis=1))

# Plot fraud probability distribution
fig, ax = plt.subplots()
sns.histplot(df['Fraud Probability'], bins=50, kde=True, color='orange', ax=ax)
ax.set_title("Distribution of Fraud Probability")
ax.set_xlabel("Fraud Probability")
ax.set_ylabel("Count")
st.pyplot(fig)

# Button to refresh (simulate re-fetch and rerun)
if st.button("Refresh Data"):
    st.experimental_rerun()
