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

# Function to find and load the latest temp CSV file
def get_latest_temp_csv(exclude_path=None):
    """
    Find the most recently modified CSV file in the temp directory.
    Excludes the original uploaded file path.
    """
    try:
        temp_dir = tempfile.gettempdir()
        csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))
        
        # Exclude the original uploaded file
        if exclude_path:
            csv_files = [f for f in csv_files if f != exclude_path]
        
        if not csv_files:
            return None
        
        # Get the most recently modified file
        latest_file = max(csv_files, key=os.path.getmtime)
        
        # Check if file was modified in the last 60 seconds (likely from agent)
        file_time = os.path.getmtime(latest_file)
        current_time = datetime.now().timestamp()
        
        if current_time - file_time < 60:  # Modified within last minute
            return latest_file
        
        return None
    except Exception as e:
        st.error(f"Error finding temp file: {e}")
        return None

# Function to create download button for temp file
def create_download_button_for_temp_file(temp_file_path):
    """
    Creates a download button for a temp CSV file and displays it as a table.
    """
    try:
        # Read the temp file
        df = pd.read_csv(temp_file_path)
        
        if df is not None and not df.empty:
            st.markdown("---")
            st.markdown("### Modified Data Table")
            st.dataframe(df)
            
            # Read file content for download
            with open(temp_file_path, 'rb') as f:
                csv_data = f.read()
            
            st.download_button(
                label="Download Modified Data as CSV",
                data=csv_data,
                file_name=f'modified_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                help=f"Download the modified dataset ({df.shape[0]} rows x {df.shape[1]} columns)"
            )
            
            st.success(f"Dataset ready for download: {df.shape[0]} rows x {df.shape[1]} columns")
            return True
    except Exception as e:
        st.error(f"Error creating download button: {e}")
        return False

# Streamlit app configuration
st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")

st.title("📊 Data Analyst Agent")

# Initialize session state variables at the very beginning
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = None
if "original_temp_path" not in st.session_state:
    st.session_state.original_temp_path = None
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    gemini_key_input = st.text_input(
        "Enter your Gemini API key:", 
        type="password", 
        value=st.session_state.get('gemini_key', '') or '',
        key="gemini_key_input"
    )
    if gemini_key_input:
        st.session_state.gemini_key = gemini_key_input
        st.success("API key saved!")
    else:
        st.warning("Please enter your Gemini API key to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and st.session_state.get('gemini_key'):
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Store original temp path in session state
        if st.session_state.get('original_temp_path') is None:
            st.session_state.original_temp_path = temp_path
        
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Initialize DuckDbTools
        duckdb_tools = DuckDbTools()
        
        # Load the CSV file into DuckDB as a table
        duckdb_tools.load_local_csv_to_table(
            path=temp_path,
            table="uploaded_data",
        )
        
        # Initialize the Agent with Gemini model
        data_analyst_agent = Agent(
            model=Gemini(id="gemini-2.5-flash", api_key=st.session_state.get('gemini_key')),
            tools=[duckdb_tools, PandasTools()],
            system_message=(
                "You are an expert data analyst. Use the uploaded_data table to answer user queries. "
                "When performing data cleaning, filtering, or transformation: "
                "1. Make changes using SQL statements on the uploaded_data table "
                "2. Export the modified data to a CSV file in the temp directory using: "
                "   COPY (SELECT * FROM uploaded_data) TO '/tmp/modified_data.csv' (HEADER, DELIMITER ',') "
                "3. Confirm the export was successful "
                "For simple queries (counts, averages), just provide the answer. "
                "Always export modified datasets to CSV so users can download them."
            ),
            markdown=True,
        )
        
        st.session_state.agent_initialized = True
        
        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")
        
        # Add info message about terminal output
        st.info("💡 Check your terminal for a clearer output of the agent's response")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get the response from the agent
                        response = data_analyst_agent.run(user_query)

                        # Extract the content from the response object
                        if hasattr(response, 'content'):
                            response_content = response.content
                        else:
                            response_content = str(response)

                    # Display the response in Streamlit
                    st.markdown("### Agent Response")
                    st.markdown(response_content)
                    
                    # Check for newly created temp files
                    original_path = st.session_state.get('original_temp_path', None)
                    latest_temp = get_latest_temp_csv(exclude_path=original_path)
                    
                    if latest_temp:
                        create_download_button_for_temp_file(latest_temp)
                    else:
                        # Try checking for /tmp/modified_data.csv specifically
                        if os.path.exists('/tmp/modified_data.csv'):
                            create_download_button_for_temp_file('/tmp/modified_data.csv')
                
                except Exception as e:
                    st.error(f"Error generating response from the agent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")