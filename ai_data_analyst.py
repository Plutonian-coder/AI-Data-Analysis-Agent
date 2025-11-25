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
import time

# ------------------------------
# Streamlit app configuration
# ------------------------------
st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")

# ------------------------------
# Persistent directory for working datasets
# ------------------------------
PERSIST_DIR = "persistent_data"
os.makedirs(PERSIST_DIR, exist_ok=True)

# ------------------------------
# Helper Functions
# ------------------------------
def preprocess_and_save(uploaded_file):
    """
    Read uploaded file, clean simple problems and save to a temporary csv.
    Returns (temp_path, columns, dataframe)
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, na_values=["NA", "N/A", "missing"])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        # Clean string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = (
                df[col].astype(str)
                .fillna("")
                .replace({r'"': '""'}, regex=True)
            )

        # Parse columns that look like dates
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Save to a temp file (we will promote to persistent file shortly)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = tmp.name
        # Save with quoting to avoid SQL/CSV parsing issues
        df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None


def promote_temp_to_persistent(temp_path):
    """
    Move/copy temp CSV into the persistent directory and return persistent path.
    We copy (not move) to avoid accidental loss if something goes wrong during copy.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    persistent_name = f"persistent_data_{timestamp}.csv"
    persistent_path = os.path.join(PERSIST_DIR, persistent_name)
    shutil.copy(temp_path, persistent_path)
    try:
        os.remove(temp_path)
    except Exception:
        # not fatal
        pass
    return persistent_path


def update_current_dataset(duckdb_tools):
    """
    Sync DuckDB -> persistent CSV -> Pandas -> DuckDB.
    After this returns, st.session_state.current_dataset will reflect the saved CSV.
    This function is the single place where we persist changes to disk.
    """
    try:
        persistent_path = st.session_state.persistent_file_path
        if not persistent_path:
            return None

        # 1) Export current DuckDB table to persistent CSV
        duckdb_tools.run_query(
            f"COPY (SELECT * FROM uploaded_data) TO '{persistent_path}' (HEADER, DELIMITER ',');"
        )

        # 2) Load CSV -> Pandas (this becomes the canonical dataframe for the UI)
        df = pd.read_csv(persistent_path)

        # 3) Recreate DuckDB table from the persistent CSV using read_csv_auto
        # This step ensures the DB table is always consistent with the CSV on disk.
        duckdb_tools.run_query("DROP TABLE IF EXISTS uploaded_data;")
        duckdb_tools.run_query(
            f"CREATE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{persistent_path}')"
        )

        return df

    except Exception as e:
        print("update_current_dataset error:", e)
        return None


def get_dataset_summary(df, max_rows=5):
    """Return a compact summary useful for system messages or quick checks."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "sample": df.head(max_rows).to_dict("records"),
        "null_counts": df.isnull().sum().to_dict(),
    }


def safe_run_agent(agent, user_input, max_retries=3, initial_delay=2):
    """
    Wrapper around agent.run that retries transient model errors (503s) with exponential backoff.
    Returns the agent response object on success, or None on permanent failure.
    """
    delay = initial_delay
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = agent.run(user_input, stream=False)
            return resp
        except Exception as e:
            last_exc = e
            err = str(e)
            # If transient (503 / Service Unavailable), retry
            if "503" in err or "Service Unavailable" in err or "temporarily unavailable" in err.lower():
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    # exhausted retries
                    return None
            else:
                # Non-transient error -> don't retry
                raise
    # If we exit loop, return None
    return None


# ------------------------------
# Session State Init
# ------------------------------
for key in [
    "gemini_key",
    "chat_history",
    "current_dataset",
    "duckdb_tools",
    "agent",
    "persistent_file_path",
    "last_uploaded_file",
]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []


# ------------------------------
# Sidebar / Config
# ------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # API key entry
    gemini_key_input = st.text_input(
        "Gemini API Key", type="password", value=st.session_state.get("gemini_key", "") or ""
    )
    if gemini_key_input:
        st.session_state.gemini_key = gemini_key_input
        st.success("Key saved!")

    st.divider()

    # Allow selection of model (default to stable 1.5)
    model_choice = st.selectbox(
        "Gemini model (choose stable)", ["gemini-1.5-flash", "gemini-2.5-flash"],
        index=0
    )
    st.caption("If you face frequent 503 errors use gemini-1.5-flash")

    st.divider()

    uploaded_file = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"])

    if uploaded_file and st.session_state.gemini_key:
        # make a file identifier to detect new uploads vs re-uploads
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_uploaded_file != file_identifier:
            # new upload - clear previous persistent file if exists
            if st.session_state.persistent_file_path and os.path.exists(st.session_state.persistent_file_path):
                try:
                    os.remove(st.session_state.persistent_file_path)
                except Exception:
                    pass
            st.session_state.persistent_file_path = None
            st.session_state.last_uploaded_file = file_identifier

        temp_path, cols, df = preprocess_and_save(uploaded_file)
        if df is not None and temp_path:
            # Promote the temp CSV into our persistent folder and set as working file
            persistent_path = promote_temp_to_persistent(temp_path)
            st.session_state.persistent_file_path = persistent_path
            st.session_state.current_dataset = df

            # Initialize DuckDB tools if necessary
            if st.session_state.duckdb_tools is None:
                st.session_state.duckdb_tools = DuckDbTools()

            # Create table from persistent CSV (only once on upload)
            try:
                st.session_state.duckdb_tools.run_query("DROP TABLE IF EXISTS uploaded_data;")
                st.session_state.duckdb_tools.run_query(
                    f"CREATE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{st.session_state.persistent_file_path}')"
                )
            except Exception as e:
                st.error(f"Error creating initial uploaded_data table: {e}")

            # Initialize agent using the selected model
            try:
                st.session_state.agent = Agent(
                    model=Gemini(id=model_choice, api_key=st.session_state.gemini_key),
                    tools=[st.session_state.duckdb_tools, PandasTools()],
                    system_message=(
                        "You are a data analyst working with a DuckDB table named 'uploaded_data'.\n"
                        "RULES:\n"
                        "1) Always run SQL against the 'uploaded_data' table for modifications.\n"
                        "2) Use statements like: CREATE OR REPLACE TABLE uploaded_data AS SELECT ...\n"
                        "3) Do NOT attempt to recreate the table from a CSV path or rely on filesystem paths.\n"
                        "4) If you modify the dataset, respond: 'I have updated the dataset' and show the SQL used.\n"
                        "5) For informational queries return summaries, not full tables.\n"
                    ),
                    markdown=True,
                )
            except Exception as e:
                st.error(f"Error initializing Agent: {e}")

            st.success("✅ File uploaded and agent initialized.")

            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # Sidebar: show dataset info if loaded
    if st.session_state.current_dataset is not None:
        st.divider()
        st.subheader("📊 Current Dataset Info")
        ds = st.session_state.current_dataset
        st.metric("Rows", ds.shape[0])
        st.metric("Columns", ds.shape[1])
        with st.expander("📋 Column Names"):
            for c in ds.columns:
                st.text(f"• {c}")

        if st.session_state.persistent_file_path and os.path.exists(st.session_state.persistent_file_path):
            st.divider()
            st.subheader("💾 Persistent Working File")
            st.caption(f"File: {os.path.basename(st.session_state.persistent_file_path)}")
            try:
                with open(st.session_state.persistent_file_path, "rb") as f:
                    data_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download working file",
                        data=data_bytes,
                        file_name=f"working_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_working"
                    )
            except Exception as e:
                st.error(f"Could not read persistent file: {e}")


# ------------------------------
# Main UI
# ------------------------------
st.title("📊 Data Analyst Agent")

if st.session_state.current_dataset is not None:
    with st.expander("👁️ View Current Dataset", expanded=False):
        st.dataframe(st.session_state.current_dataset, use_container_width=True)

st.divider()

# Render chat history
for i, msg in enumerate(st.session_state.chat_history):
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            if msg.get("dataset_modified") and msg.get("modified_data") is not None:
                mod_df = msg["modified_data"]
                st.divider()
                st.success(f"✅ Dataset Modified: {mod_df.shape[0]} rows × {mod_df.shape[1]} columns")
                # show only first 100 rows for performance
                show_df = mod_df.head(100) if len(mod_df) > 100 else mod_df
                if len(mod_df) > 100:
                    st.info(f"Showing first 100 rows of {len(mod_df)} total rows")
                st.dataframe(show_df, use_container_width=True)
                csv_bytes = mod_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Modified Dataset",
                    data=csv_bytes,
                    file_name=f"modified_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"dl_mod_{i}"
                )

st.divider()

# Chat input / main loop
if st.session_state.current_dataset is not None and st.session_state.gemini_key and st.session_state.agent:
    user_input = st.chat_input("Ask a question or request a modification...")
    if user_input:
        # Save user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🤔 Analyzing..."):
            # Run the agent with safe wrapper (retries on 503)
            response = safe_run_agent(st.session_state.agent, user_input, max_retries=3, initial_delay=2)

            if response is None:
                # Transient failure -> inform user, do not crash; keep UI responsive
                error_text = "⚠️ The language model is temporarily unavailable (503). Please try again shortly or switch to a different model."
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_text,
                    "dataset_modified": False,
                    "modified_data": None,
                })
            else:
                # Normal response path
                try:
                    response_content = response.content if hasattr(response, "content") else str(response)
                except Exception:
                    response_content = str(response)

                # Persist current DB table to disk and reload into pandas (safely)
                updated_df = update_current_dataset(st.session_state.duckdb_tools)

                dataset_modified = False
                if updated_df is not None and st.session_state.current_dataset is not None:
                    if not updated_df.equals(st.session_state.current_dataset):
                        dataset_modified = True
                        st.session_state.current_dataset = updated_df

                # Append assistant message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_content,
                    "dataset_modified": dataset_modified,
                    "modified_data": updated_df if dataset_modified else None
                })

        # Re-run so UI updates
        st.rerun()

elif st.session_state.current_dataset is None:
    st.info("👈 Please upload a CSV or Excel file from the sidebar to get started.")
elif not st.session_state.gemini_key:
    st.info("👈 Please enter your Gemini API key in the sidebar to get started.")
else:
    st.info("Agent not initialized - ensure API key and uploaded file are present.")

st.divider()
st.caption("💡 All dataset changes are saved to the persistent working file and used for subsequent queries.")
st.caption("⚡ Keep queries specific and concise for best results.")
