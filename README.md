# FinInsight — An Intelligent Financial Chatbot

FinInsight (also referred to as FinSight in the codebase) is an educational financial-research prototype designed to bridge the gap between qualitative management commentary and quantitative financial indicators. It processes unstructured earnings-call transcripts and structured SEC financial datasets to curate an evidence-based knowledge layer. 

**Academic Context:** This project was developed by B. Shabareshwar Reddy, Nayanojwal Sri Tej, and Sreekar Bharadwaz (Vellore Institute of Technology, Chennai) for CSE4074 – Forensic Analytics.

> **Note:** For a comprehensive dive into the methodology, architectural diagrams, data pipeline evaluations, and experimental models, please refer to the detailed project report included in this repository: **FinInsight_Project_Report_with_Figures.docx**.

> **Disclaimer:** FinInsight is an academic/educational prototype. It does not provide investment, legal, tax, or accounting advice.

---

## 🛠 Technologies Used & How We Used Them

### 1. Document Extraction & Preprocessing
*   **Python:** Core programming language for all pipelines and application logic.
*   **PyMuPDF (fitz):** Extracts raw text from earnings-call and prepared-remarks PDF files.
*   **Regex:** Powers our **Speaker-Aware Dynamic Chunking (HSADC)** to separate transcripts into speaker turns (e.g., CEO, Analyst) and extract raw financial metrics.
*   **NLTK (punkt):** Handles precise sentence segmentation within individual speaker chunks.

### 2. Natural Language Processing (NLP) & Classification
*   **Hugging Face Transformers:** Framework for specialized NLP models.
*   **ProsusAI/finbert:** Classifies financial sentiment (Positive, Negative, Neutral) and generates confidence scores.
*   **facebook/bart-large-mnli:** Zero-shot classifier for two tasks:
    1.  **Forward-Guidance Classification:** (e.g., *Revenue Guidance*, *Strategic Business Outlook*).
    2.  **Risk Classification:** (e.g., *Macroeconomic Uncertainty*, *Competitive Pressure*).

### 3. Data Processing & Structured Analytics
*   **pandas & NumPy:** Processes U.S. SEC Financial Statement Data Sets (2025 Q4), filters XBRL tags, and structures `financial_facts.csv`.
*   **pymannkendall:** Runs the Mann-Kendall trend test on extracted metrics.

### 4. Experimental Decision Modeling
*   **TOPSIS (via NumPy):** Experimental Multi-Criteria Decision Analysis (MCDA) ranking system based on FinBERT confidence and guidance intensity.
*   **XGBoost & scikit-learn:** Experimental `XGBRegressor` prototype to explore feature-target relationships (trained on a simulated target variable for workflow validation).

### 5. Conversational Interface (Chatbot Prototype)
*   **Streamlit:** Interactive web UI for company selection and natural language Q&A.
*   **OpenAI Python Client:** Facilitates API requests to the LLM.
*   **NVIDIA Hosted LLM Endpoint (`nemotron-3.5-lightning-30b-a3b`):** Generates evidence-grounded, cited responses from compiled SEC facts and transcript contexts.

---

## ⚙️ System Workflow Summary

1.  **Transcript Pipeline:** Reads PDFs $\rightarrow$ Chunks by Speaker (Regex) $\rightarrow$ Segments to Sentences (NLTK) $\rightarrow$ Rule-based filtering $\rightarrow$ Sentiment Analysis (FinBERT) $\rightarrow$ Zero-Shot Classification (BART).
2.  **Financial Pipeline:** Ingests SEC Data Sets $\rightarrow$ Filters Exact Tags/Consolidated Values (pandas) $\rightarrow$ Aligns Accession Numbers & Periods.
3.  **Retrieval-Augmented Generation (RAG):** Retrieves the most recent SEC metrics and performs keyword-overlap searches against categorized transcript sentences.
4.  **Generation:** Context (facts + quotes + metadata) is passed to the LLM with strict system prompts to cite sources and refuse to guess missing data.

---

## 🚀 Getting Started (Local Setup)

### 1. Requirements
*   **Python:** 3.12 recommended.
*   **API Key:** An NVIDIA API key for the Nemotron LLM endpoint.

### 2. Installation
Create and activate a virtual environment, then install dependencies:
```bash
# Using uv (or standard venv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Configure API Key
Set your NVIDIA API key as an environment variable (or enter it in the app UI):
```bash
# macOS/Linux
export NVIDIA_API_KEY="your-api-key"
export NVIDIA_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b"

# Windows (PowerShell)
$env:NVIDIA_API_KEY = "your-api-key"
$env:NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
```
*(Alternatively, use Streamlit secrets by creating `.streamlit/secrets.toml`)*

### 4. Run the App
```bash
python -m streamlit run app.py
```

### 5. Run Tests
Verify data selection, retrieval context formatting, and prompt construction:
```bash
python -m unittest discover -s tests -v
```

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push your repository to GitHub (ensure `requirements.txt` and the `data/` folder are included). **Do not commit API keys.**
2. Log in to Streamlit Community Cloud and click "New app".
3. Select your repository, branch, and set the main file path to `app.py`.
4. Navigate to **Advanced settings > Secrets** and add your credentials:
   ```toml
   NVIDIA_API_KEY = "your-api-key"
   NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
   ```
5. Click **Deploy**.

---

## ⚠️ Known Methodological Limitations
*   **Retrieval Mechanism:** The chatbot uses keyword-overlap retrieval rather than dense semantic embeddings; sentences without exact keyword matches may be overlooked.
*   **Extraction Brittleness:** Regex-based speaker chunking depends heavily on specific PDF formatting, causing parsing failures on non-standard transcripts.
*   **Experimental Scope:** The TOPSIS ranking system and XGBoost predictor are exploratory analytical experiments and are **not** integrated into the active chatbot logic.
