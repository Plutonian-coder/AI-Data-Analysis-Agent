import tempfile
import csv
import streamlit as st
import pandas as pd
from agno.agent import Agent
from agno.models.google import Gemini 
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools
import os
from datetime import datetime
import time

# Streamlit app configuration
st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")

# --- Helper Functions ---

def preprocess_and_save(file):
    """Reads uploaded file, cleans it, and saves to temp CSV."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Clean string columns: remove quotes that might break SQL
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).fillna('').replace({r'"': '""'}, regex=True)
        
        # Parse dates
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Save to temp file with specific quoting
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

def update_current_dataset(duckdb_tools):
    """Exports the current state of 'uploaded_data' table back to session state and saves permanently."""
    try:
        # Use a persistent file path stored in session state
        if 'persistent_file_path' not in st.session_state or st.session_state.persistent_file_path is None:
            # Create a persistent file path for this session
            st.session_state.persistent_file_path = f'{tempfile.gettempdir()}/persistent_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        persistent_path = st.session_state.persistent_file_path
        
        # Run a direct SQL COPY command to export the data to persistent file
        if hasattr(duckdb_tools, 'connection'):
            duckdb_tools.connection.execute(f"COPY (SELECT * FROM uploaded_data) TO '{persistent_path}' (HEADER, DELIMITER ',')")
        else:
            # Fallback for different library versions
            duckdb_tools.run_query(f"COPY (SELECT * FROM uploaded_data) TO '{persistent_path}' (HEADER, DELIMITER ',')")

        # Read the exported file back into Pandas
        if os.path.exists(persistent_path):
            df = pd.read_csv(persistent_path)
            return df
        return None
    except Exception as e:
        print(f"Update warning: {str(e)}")
        return None

def get_dataset_summary(df, max_rows=5):
    """Get a concise summary of the dataset to minimize token usage."""
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "sample": df.head(max_rows).to_dict('records'),
        "null_counts": df.isnull().sum().to_dict()
    }
    return summary

# --- Session State Init ---

keys = ["gemini_key", "chat_history", "current_dataset", "duckdb_tools", "agent", "persistent_file_path", "last_uploaded_file", "retry_count"]
for key in keys:
    if key not in st.session_state:
        st.session_state[key] = None
if "chat_history" not in st.session_state or st.session_state.chat_history is None:
    st.session_state.chat_history = []
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0

# --- Sidebar ---

with st.sidebar:
    st.header("⚙️ Configuration")
    
    gemini_key_input = st.text_input(
        "Gemini API Key", 
        type="password", 
        value=st.session_state.get('gemini_key', '') or ''
    )
    if gemini_key_input:
        st.session_state.gemini_key = gemini_key_input
        st.success("Key saved!")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file and st.session_state.gemini_key:
        # Check if this is a new file upload
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != file_identifier:
            # New file uploaded - clean up old persistent file if exists
            if st.session_state.persistent_file_path and os.path.exists(st.session_state.persistent_file_path):
                try:
                    os.remove(st.session_state.persistent_file_path)
                    st.info("Previous working file cleaned up.")
                except:
                    pass
            
            # Reset persistent file path for new upload
            st.session_state.persistent_file_path = None
            st.session_state.last_uploaded_file = file_identifier
        
        temp_path, columns, df = preprocess_and_save(uploaded_file)
        
        if df is not None:
            st.session_state.current_dataset = df
            
            if st.session_state.duckdb_tools is None:
                st.session_state.duckdb_tools = DuckDbTools()
            
            # Create persistent file path if not exists
            if st.session_state.persistent_file_path is None:
                st.session_state.persistent_file_path = f'{tempfile.gettempdir()}/persistent_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            # Save to persistent location (this becomes the working file)
            df.to_csv(st.session_state.persistent_file_path, index=False, quoting=csv.QUOTE_ALL)
            
            # Delete the original temp file since we now have persistent file
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            
            # Create Table in DuckDB (Using persistent file)
            try:
                query = f"CREATE OR REPLACE TABLE uploaded_data AS SELECT * FROM '{st.session_state.persistent_file_path}'"
                st.session_state.duckdb_tools.run_query(query)
            except Exception as e:
                st.error(f"Error creating table: {e}")
            
            # Initialize Agent with optimized settings
            st.session_state.agent = Agent(
                model=Gemini(
                    id="gemini-2.0-flash-exp",  # Using faster, more stable model
                    api_key=st.session_state.gemini_key
                ),
                tools=[st.session_state.duckdb_tools, PandasTools()],
                system_message=(
                    "You are a data analyst working with a table called 'uploaded_data' in DuckDB. "
                    "IMPORTANT INSTRUCTIONS:\n"
                    "1. Keep responses concise and focused.\n"
                    "2. Use SQL queries on 'uploaded_data' table for modifications.\n"
                    "3. For filtering/cleaning: 'CREATE OR REPLACE TABLE uploaded_data AS SELECT ...'\n"
                    "4. For simple questions, provide brief answers without unnecessary details.\n"
                    "5. If you modify data, say 'I have updated the dataset' and show the operation.\n"
                    "6. Avoid returning entire datasets in your response - only summaries.\n"
                    f"Current dataset info: {df.shape[0]} rows, {df.shape[1]} columns\n"
                    f"Columns: {', '.join(df.columns.tolist())}"
                ),
                markdown=True,
                show_tool_calls=False,  # Reduce output verbosity
                add_history_to_messages=True,
                num_history_responses=3,  # Limit history to last 3 interactions
            )
            st.success("✅ Agent Ready")
            
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.retry_count = 0
                st.rerun()

    # --- Sidebar Dataset Info ---
    if st.session_state.current_dataset is not None:
        st.divider()
        st.subheader("📊 Current Dataset Info")
        df = st.session_state.current_dataset
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        
        with st.expander("📋 Column Names"):
            for col in df.columns:
                st.text(f"• {col}")
        
        # Show persistent file location and download button
        if st.session_state.persistent_file_path and os.path.exists(st.session_state.persistent_file_path):
            st.divider()
            st.subheader("💾 Persistent File")
            st.caption(f"Working file: {os.path.basename(st.session_state.persistent_file_path)}")
            st.caption("All changes are saved to this file automatically.")
            
            # Download button for persistent file
            try:
                with open(st.session_state.persistent_file_path, 'rb') as f:
                    persistent_data = f.read()
                    st.download_button(
                        label="⬇️ Download Current Working File",
                        data=persistent_data,
                        file_name=f"working_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',
                        key="download_persistent_sidebar"
                    )
            except Exception as e:
                st.error(f"Error reading persistent file: {e}")


# --- Main Logic ---

st.title("📊 Data Analyst Agent")

# Display current dataset preview (Global View)
if st.session_state.current_dataset is not None:
    with st.expander("👁️ View Current Dataset", expanded=False):
        st.dataframe(st.session_state.current_dataset, use_container_width=True)

# Chat interface
st.divider()

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            
            # Check for dataset modification and render inline preview/download
            if message.get("dataset_modified") and message.get("modified_data") is not None:
                modified_df = message["modified_data"]
                
                st.divider()
                st.success(f"✅ Dataset Modified: {modified_df.shape[0]} rows × {modified_df.shape[1]} columns")
                
                # Show modified dataset in table (limit rows for performance)
                display_df = modified_df.head(100) if len(modified_df) > 100 else modified_df
                if len(modified_df) > 100:
                    st.info(f"Showing first 100 rows of {len(modified_df)} total rows")
                
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv_data = modified_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Modified Dataset",
                    data=csv_data,
                    file_name=f'modified_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    key=f"download_{i}_{datetime.now().strftime('%f')}"
                )

# Chat input
if st.session_state.current_dataset is not None and st.session_state.get('gemini_key'):
    user_input = st.chat_input("Ask a question or request a modification...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get agent response with retry logic
        max_retries = 3
        retry_delay = 2
        
        with st.spinner("🤔 Analyzing..."):
            for attempt in range(max_retries):
                try:
                    # Run the agent
                    response = st.session_state.agent.run(user_input, stream=False)
                    
                    # Extract response content
                    response_content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Limit response length to prevent token overflow
                    if len(response_content) > 5000:
                        response_content = response_content[:5000] + "\n\n... (response truncated for brevity)"
                    
                    # Update the current dataset from DuckDB
                    updated_df = update_current_dataset(st.session_state.duckdb_tools)
                    
                    # Check if dataset was actually modified
                    dataset_modified = False
                    if updated_df is not None and st.session_state.current_dataset is not None:
                        # Compare with previous state
                        if not updated_df.equals(st.session_state.current_dataset):
                            dataset_modified = True
                            st.session_state.current_dataset = updated_df
                            
                            # CRITICAL: Reload the updated data back into DuckDB
                            # This ensures the next query works on the modified dataset
                            try:
                                reload_query = f"CREATE OR REPLACE TABLE uploaded_data AS SELECT * FROM '{st.session_state.persistent_file_path}'"
                                st.session_state.duckdb_tools.run_query(reload_query)
                            except Exception as reload_error:
                                print(f"Reload warning: {str(reload_error)}")
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_content,
                        "dataset_modified": dataset_modified,
                        "modified_data": updated_df if dataset_modified else None
                    })
                    
                    # Reset retry count on success
                    st.session_state.retry_count = 0
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check for specific API errors
                    if "503" in error_str or "Service Unavailable" in error_str:
                        if attempt < max_retries - 1:
                            st.warning(f"API temporarily unavailable. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            error_msg = "❌ Error: The Gemini API is currently unavailable. Please try again in a few moments."
                    elif "429" in error_str or "rate limit" in error_str.lower():
                        error_msg = "❌ Error: Rate limit exceeded. Please wait a moment before trying again."
                    elif "quota" in error_str.lower():
                        error_msg = "❌ Error: API quota exceeded. Please check your Gemini API usage."
                    elif "timeout" in error_str.lower():
                        error_msg = "❌ Error: Request timed out. The query might be too complex. Try simplifying it."
                    else:
                        error_msg = f"❌ Error: {error_str}"
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "dataset_modified": False,
                        "modified_data": None
                    })
                    break
        
        # Rerun to display the updated chat
        st.rerun()

elif st.session_state.current_dataset is None:
    st.info("👈 Please upload a CSV or Excel file from the sidebar to get started.")
elif not st.session_state.get('gemini_key'):
    st.info("👈 Please enter your Gemini API key in the sidebar to get started.")

# Footer
st.divider()
st.caption("💡 Tip: Each query works on the current state of your dataset. All modifications persist throughout the session.")
st.caption("⚡ For best performance, keep queries specific and concise.")