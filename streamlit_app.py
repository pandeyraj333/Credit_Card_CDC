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

    feature_columns = joblib.load("feature_columns.joblib")

    # Add any missing dummy columns and sort as in training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    return df[feature_columns]


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
    select * from credit_card_transaction_data order by _peerdb_synced_at desc
    """

    result = client.query(query)
    df_fetched = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df_fetched.copy()

def load_unpredicted():
    query = '''select * from credit_card_predictions where predicted_value is NULL'''
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df
def make_and_push_predictions(df):
    if df.empty:
        st.success("All data already has predictions!")
        return

    X = preprocess(df)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    # 3. Insert predictions back using a batch insert
    df['predicted_value'] = preds

    # You can insert this to another table, or update the main table
    # Here, for a new predictions table:
    # columns_to_write = ['row_id', 'predicted_value']  # or your PK/UID column + prediction
    # data_to_write = df[columns_to_write].to_dict('records')
    # client.insert('credit_card_predictions', data_to_write, column_names=columns_to_write)

    # OR to update the existing table:
    for idx, row in df.iterrows():
        query = f"ALTER TABLE credit_card_predictions UPDATE predicted_value = {int(row['predicted_value'])} WHERE nameOrig = '{row['nameOrig']} and nameDest = '{row['nameDest']}'"
        client.command(query)

    st.success(f"Pushed predictions for {len(df)} rows.")


st.title("Credit Card Transaction Fraud Detection")

# Fetch and preprocess data
df_main = load_clickhouse_data()
st.title("Credit Card Transactions Overview")

total_txns = len(df_main)
fraud_rate = 0.0
if 'isFlaggedFraud' in df_main.columns:
    fraud_rate = df_main['isFlaggedFraud'].mean() * 100

st.metric("Total Transactions", total_txns)
st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")


st.metric("Total Transactions", total_txns)
st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

# Visual - Pie chart of fraud vs non-fraud
fraud_counts = df_main['isFlaggedFraud'].value_counts().sort_index()
labels = ['Not Fraud', 'Fraud']
sizes = [fraud_counts.get(0, 0), fraud_counts.get(1, 0)]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
ax.axis('equal')  # Equal aspect ratio ensures pie is circular
ax.set_title("Fraud vs Non-Fraud Transactions")

st.pyplot(fig)


# # Interactive filter by prediction
# filter_option = st.selectbox("Filter transactions by prediction:",  ["All", "Not Fraud", "Fraud"])

# if filter_option == "Not Fraud":
#     df_filtered = df[df['Predicted Fraud'] == 0]
# elif filter_option == "Fraud":
#     df_filtered = df[df['Predicted Fraud'] == 1]
# else:
#     df_filtered = df

# Show table with highlighting fraud rows
# def highlight_fraud(row):
#     return ['background-color: #f87171' if v == 1 else '' for v in row['Predicted Fraud':'Predicted Fraud']]

# # We highlight rows entirely for simplicity here on predicted fraud
# def highlight_row(row):
#     if row['Predicted Fraud'] == 1:
#         return ['background-color: #fddede'] * len(row)
#     else:
#         return [''] * len(row)

# st.write(f"Showing {len(df_filtered)} transactions")



# Button to refresh (simulate re-fetch and rerun)
if st.button("Fetch Unpredicted"):
    df_unpredicted = load_unpredicted()
    st.dataframe(df_unpredicted)
    st.write(f"{len(df_unpredicted)} transactions need prediction.")
if st.button("Push Missing Predictions"):
    if len(df_unpredicted) > 0:
        make_and_push_predictions(df_unpredicted) 
