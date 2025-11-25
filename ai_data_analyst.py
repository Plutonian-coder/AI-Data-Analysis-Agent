import tempfile
import csv
import streamlit as st
import pandas as pd
from agno.agent import Agent
from agno.models.google import Gemini 
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools
import os
import glob
from datetime import datetime

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).fillna('').replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError):
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Function to update the current dataset in session and DuckDB
def update_current_dataset(duckdb_tools):
    """
    Updates the current dataset in session state from DuckDB.
    This keeps the working dataset in sync after modifications.
    """
    try:
        # Export current state from DuckDB to temp file
        temp_export_path = f'/tmp/current_session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        duckdb_tools.run_query(f"COPY (SELECT * FROM uploaded_data) TO '{temp_export_path}' (HEADER, DELIMITER ',')")
        
        # Read it back into session state
        df = pd.read_csv(temp_export_path)
        st.session_state.current_dataset = df
        st.session_state.current_dataset_path = temp_export_path
        
        return df, temp_export_path
    except Exception as e:
        st.error(f"Error updating dataset: {e}")
        return None, None

# Function to create download button for current dataset
def create_download_button(df):
    """
    Creates a download button for the current working dataset.
    """
    try:
        if df is not None and not df.empty:
            # Prepare CSV data
            csv_buffer = pd.io.common.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="⬇️ Download Current Dataset",
                data=csv_data,
                file_name=f'modified_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                help=f"Download the current working dataset ({df.shape[0]} rows × {df.shape[1]} columns)",
                key=f"download_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            return True
    except Exception as e:
        st.error(f"Error creating download button: {e}")
        return False

# Streamlit app configuration
st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")

# Initialize session state variables
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = None
if "current_dataset_path" not in st.session_state:
    st.session_state.current_dataset_path = None
if "original_temp_path" not in st.session_state:
    st.session_state.original_temp_path = None
if "duckdb_tools" not in st.session_state:
    st.session_state.duckdb_tools = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "columns" not in st.session_state:
    st.session_state.columns = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    gemini_key_input = st.text_input(
        "Enter your Gemini API key:", 
        type="password", 
        value=st.session_state.get('gemini_key', '') or '',
        key="gemini_key_input"
    )
    if gemini_key_input:
        st.session_state.gemini_key = gemini_key_input
        st.success("✅ API key saved!")
    else:
        st.warning("⚠️ Please enter your Gemini API key to proceed.")
    
    st.divider()
    
    # File Upload
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None and st.session_state.get('gemini_key'):
        # Preprocess and save the uploaded file
        temp_path, columns, df = preprocess_and_save(uploaded_file)
        
        if temp_path and columns and df is not None:
            # Store in session state
            st.session_state.original_temp_path = temp_path
            st.session_state.current_dataset = df
            st.session_state.current_dataset_path = temp_path
            st.session_state.columns = columns
            
            # Initialize DuckDB if not already done
            if st.session_state.duckdb_tools is None:
                st.session_state.duckdb_tools = DuckDbTools()
            
            # Load into DuckDB
            st.session_state.duckdb_tools.load_local_csv_to_table(
                path=temp_path,
                table="uploaded_data",
            )
            
            # Initialize Agent if not already done
            if st.session_state.agent is None:
                st.session_state.agent = Agent(
                    model=Gemini(id="gemini-2.5-flash", api_key=st.session_state.get('gemini_key')),
                    tools=[st.session_state.duckdb_tools, PandasTools()],
                    system_message=(
                        "You are an expert data analyst. Use the uploaded_data table to answer user queries. "
                        "When performing data cleaning, filtering, or transformation: "
                        "1. Make ALL changes directly on the uploaded_data table using SQL UPDATE, DELETE, or INSERT statements "
                        "2. For filtering/subsetting, use: CREATE OR REPLACE TABLE uploaded_data AS SELECT ... FROM uploaded_data WHERE ... "
                        "3. After making changes, confirm what was done "
                        "4. DO NOT create new tables - always modify uploaded_data in place "
                        "For simple queries (counts, averages), just provide the answer. "
                        "Remember: All modifications should persist in the uploaded_data table for the next query."
                    ),
                    markdown=True,
                )
            
            st.success(f"✅ File loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Clear chat history on new upload
            if st.button("🔄 Reset Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    st.divider()
    
    # Dataset Info
    if st.session_state.current_dataset is not None:
        st.subheader("📊 Current Dataset Info")
        df = st.session_state.current_dataset
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        
        with st.expander("📋 Column Names"):
            for col in df.columns:
                st.text(f"• {col}")
        
        # Download button in sidebar
        st.divider()
        create_download_button(df)

# Main content area
st.title("📊 Data Analyst Agent")
st.markdown("Ask questions about your data or request modifications in natural language.")

# Display current dataset preview
if st.session_state.current_dataset is not None:
    with st.expander("👁️ View Current Dataset", expanded=False):
        st.dataframe(st.session_state.current_dataset, use_container_width=True)

# Chat interface
st.divider()

# Display chat history
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                # Show dataset preview if it was modified
                if message.get("dataset_modified") and message.get("modified_data") is not None:
                    with st.expander("📊 Modified Dataset Preview"):
                        st.dataframe(message["modified_data"], use_container_width=True)
                        st.caption(f"Shape: {message['modified_data'].shape[0]} rows × {message['modified_data'].shape[1]} columns")

# Chat input
if st.session_state.current_dataset is not None and st.session_state.get('gemini_key'):
    user_input = st.chat_input("Ask a question or request a modification...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Run the agent
                    response = st.session_state.agent.run(user_input)
                    
                    # Extract response content
                    if hasattr(response, 'content'):
                        response_content = response.content
                    else:
                        response_content = str(response)
                    
                    # Display response
                    st.markdown(response_content)
                    
                    # Update the current dataset from DuckDB
                    updated_df, updated_path = update_current_dataset(st.session_state.duckdb_tools)
                    
                    # Check if dataset was actually modified
                    dataset_modified = False
                    if updated_df is not None:
                        # Compare with previous state
                        if not updated_df.equals(st.session_state.current_dataset):
                            dataset_modified = True
                            st.session_state.current_dataset = updated_df
                            st.session_state.current_dataset_path = updated_path
                            
                            # Show preview of modified data
                            with st.expander("📊 Modified Dataset Preview"):
                                st.dataframe(updated_df, use_container_width=True)
                                st.caption(f"Shape: {updated_df.shape[0]} rows × {updated_df.shape[1]} columns")
                            
                            st.success(f"✅ Dataset updated: {updated_df.shape[0]} rows × {updated_df.shape[1]} columns")
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_content,
                        "dataset_modified": dataset_modified,
                        "modified_data": updated_df if dataset_modified else None
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "dataset_modified": False,
                        "modified_data": None
                    })
        
        # Rerun to update the chat display
        st.rerun()

elif st.session_state.current_dataset is None:
    st.info("👈 Please upload a CSV or Excel file from the sidebar to get started.")
elif not st.session_state.get('gemini_key'):
    st.info("👈 Please enter your Gemini API key in the sidebar to get started.")

# Footer
st.divider()
st.caption("💡 Tip: Each query works on the current state of your dataset. All modifications persist throughout the session.")