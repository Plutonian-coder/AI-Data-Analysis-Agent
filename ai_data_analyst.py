import tempfile
import csv
import streamlit as st
import pandas as pd
from agno.agent import Agent
from agno.models.google import Gemini 
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools
from datetime import datetime
import os
import shutil

# ---------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------
st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")


# ---------------------------------------------------------
# Ensure a permanent directory for persistent CSVs
# ---------------------------------------------------------
PERSIST_DIR = "persistent_data"

def startup():
    """Clear the persistent data directory on startup."""
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    os.makedirs(PERSIST_DIR, exist_ok=True)

startup()


# ---------------------------------------------------------
# Helper: Load file cleanly and save as a persistent CSV
# ---------------------------------------------------------
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported format. Upload CSV or Excel.")
            return None, None, None
        
        # Clean object columns
        for col in df.select_dtypes(include=['object']):
            df[col] = (
                df[col]
                .astype(str)
                .fillna('')
                .replace({r'"': '""'}, regex=True)
            )

        # Parse dates
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='ignore')

        # Create temp CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None


# ---------------------------------------------------------
# Helper: Sync DuckDB → Persistent CSV → Pandas → DuckDB
# ---------------------------------------------------------
def update_current_dataset(duckdb_tools):
    try:
        persistent_path = st.session_state.persistent_file_path

        # 1. DuckDB → CSV (persistent)
        duckdb_tools.run_query(
            f"COPY (SELECT * FROM uploaded_data) "
            f"TO '{persistent_path}' (HEADER, DELIMITER ',');"
        )

        # 2. Load CSV → Pandas
        df = pd.read_csv(persistent_path)

        # 3. Load Pandas → DuckDB
        duckdb_tools.run_query("DROP TABLE IF EXISTS uploaded_data;")
        duckdb_tools.run_query(
            f"CREATE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{persistent_path}')"
        )

        return df

    except Exception as e:
        print("Update error:", e)
        return None


# ---------------------------------------------------------
# Session State Init
# ---------------------------------------------------------
for key in ["gemini_key", "chat_history", "current_dataset", "duckdb_tools", "agent", "persistent_file_path"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Gemini key
    gemini_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get("gemini_key", "") or ""
    )
    if gemini_key_input:
        st.session_state.gemini_key = gemini_key_input
        st.success("Key saved!")

    st.divider()

    # File upload
    uploaded_file = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"])

    if uploaded_file and st.session_state.gemini_key:
        # Reset agent and chat history on new file upload
        st.session_state.agent = None
        st.session_state.chat_history = []
        st.session_state.duckdb_tools = None

        temp_path, columns, df = preprocess_and_save(uploaded_file)

        if df is not None:
            st.session_state.current_dataset = df

            # Create a fixed path for the persistent file
            persistent_path = os.path.join(PERSIST_DIR, "persistent_data.csv")
            st.session_state.persistent_file_path = persistent_path

            # Save CSV permanently, overwriting if it exists
            shutil.copy(temp_path, persistent_path)

            # Initialize DuckDB
            st.session_state.duckdb_tools = DuckDbTools()

            # Create table from persistent CSV
            st.session_state.duckdb_tools.run_query("DROP TABLE IF EXISTS uploaded_data;")
            st.session_state.duckdb_tools.run_query(
                f"CREATE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{persistent_path}')"
            )

            # Initialize Agent
            st.session_state.agent = Agent(
                model=Gemini(id="gemini-2.5-flash", api_key=st.session_state.gemini_key),
                tools=[st.session_state.duckdb_tools, PandasTools()],
                system_message=(
                    "You are a data analyst.\n"
                    "The table 'uploaded_data' is already loaded.\n"
                    "When modifying data, use SQL directly on 'uploaded_data'.\n"
                    "Example: CREATE OR REPLACE TABLE uploaded_data AS SELECT ...\n"
                    "Never recreate the table from a CSV file.\n"
                    "If you modify the dataset, say 'I have updated the dataset'."
                ),
                markdown=True,
            )

            st.success("✅ Agent Ready!")

            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()


    # Dataset info
    if st.session_state.current_dataset is not None:
        st.divider()
        st.subheader("📊 Current Dataset Info")
        df = st.session_state.current_dataset
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])

        with st.expander("📋 Columns"):
            for col in df.columns:
                st.text(f"• {col}")

        if st.session_state.persistent_file_path:
            with st.expander("💾 File Info"):
                st.caption(f"Saved CSV: {os.path.basename(st.session_state.persistent_file_path)}")


# ---------------------------------------------------------
# Main Area
# ---------------------------------------------------------
st.title("📊 Data Analyst Agent")

# Dataset preview
if st.session_state.current_dataset is not None:
    with st.expander("👁️ View Current Dataset", expanded=False):
        st.dataframe(st.session_state.current_dataset, use_container_width=True)

st.divider()

# Chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

            if message.get("dataset_modified"):
                df = message["modified_data"]

                st.success(f"✅ Dataset Modified: {df.shape[0]} rows × {df.shape[1]} columns")
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Modified Dataset",
                    data=csv_data,
                    file_name=f"modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_{i}"
                )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------
if st.session_state.current_dataset is not None and st.session_state.gemini_key:
    user_input = st.chat_input("Ask a question or modify the dataset...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            response = st.session_state.agent.run(user_input, stream=False)
            response_content = response.content

            updated_df = update_current_dataset(st.session_state.duckdb_tools)

            dataset_modified = False
            if updated_df is not None and not updated_df.equals(st.session_state.current_dataset):
                dataset_modified = True
                st.session_state.current_dataset = updated_df

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_content,
                "dataset_modified": dataset_modified,
                "modified_data": updated_df if dataset_modified else None
            })

        st.rerun()


# ---------------------------------------------------------
st.divider()
st.caption("💡 All dataset changes persist across the session.")
