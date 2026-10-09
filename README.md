# 🛡️ Clinical R Code Compliance Agent

An offline-first, AI-powered compliance agent designed specifically for clinical R programming (SDTM/ADaM). 

This tool automates the tedious code review process by analyzing R scripts against clinical programming standards. Crucially, it utilizes **100% local, self-contained Large Language Models (LLMs)**, ensuring strict data privacy and zero external API calls. Your proprietary clinical trial scripts and datasets never leave your machine.

---

## 🏗️ Core Architecture

The agent operates on a sophisticated three-stage pipeline to guarantee deterministic reliability alongside AI reasoning:

1. **Deterministic Analysis**: Fast, regex-based parsing that scans the AST/code text to flag potential violations and pinpoint exact line numbers.
2. **Local RAG (Retrieval-Augmented Generation)**: Uses a local `ChromaDB` vector store and HuggingFace `all-MiniLM-L6-v2` embeddings to instantly retrieve the most relevant JSON-based compliance rules for the flagged code snippet.
3. **AI Agent**: Built with LangChain, the agent orchestrates a local `gemma4:e4b` model (via Ollama). It evaluates the flagged context, maps findings to strict Pydantic schemas, resolves multi-rule conflicts, and generates exact drop-in code patches.

---

## ✨ Key Features

* **Streamlit Web Dashboard**: An interactive UI for clinical analysts to upload `.R` scripts, configure concurrency, and download reports.
* **Asynchronous CLI**: A lightning-fast command-line interface that processes multiple LLM rule-checks concurrently using `asyncio`.
* **Clinical Validation Reporting**: Generates beautiful, Pinnacle 21-style HTML dashboards and structured Excel (`.xlsx`) workbooks detailing the severity, evidence, and suggested fixes for every issue.
* **Evaluation Framework**: A built-in batch evaluator measuring True Positives, False Positives, Precision, Recall, and F1-score against ground-truth datasets, consistently achieving 100% metrics on the synthetic test suite.
* **Docker Containerization**: Fully containerized environment for reproducible, isolated deployments.

---

## ⚙️ Prerequisites & Installation

To run the agent locally, you need Python 3.11+ and [Ollama](https://ollama.com/) installed on your machine.

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/clinical-r-code-compliance-agent.git
cd clinical-r-code-compliance-agent
```

**2. Set up a Python Virtual Environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Pull the Local LLM via Ollama**
```bash
ollama pull gemma4:e4b
```

---

## 🚀 Quick Start / Usage

### 1. Interactive Web Dashboard
Launch the Streamlit web interface to easily upload files and view interactive visualizations.
```bash
streamlit run app.py
```
*Navigate to `http://localhost:8501` in your browser.*

### 2. Command Line Interface (CLI)
Run a single R script through the pipeline and export the HTML/Excel reports directly to the output folder.
```bash
python scripts/cli.py examples/input/analysis_script_1.R --use-llm --export
```

### 3. Batch Evaluation
Run the automated evaluation suite against the synthetic ground-truth dataset to calculate classification metrics.
```bash
python scripts/evaluate.py
```

---

## 📂 Project Structure

```text
clinical-r-code-compliance-agent/
├── app.py                      # Streamlit interactive frontend
├── Dockerfile                  # Containerization instructions
├── evaluation/                 # Ground truth datasets & batch scripts
│   ├── ground_truth.csv
│   └── scripts/
├── examples/                   # Sample inputs and generated HTML/Excel outputs
│   ├── input/
│   └── output/
├── knowledge/                  # JSON rule definitions & vector DB storage
│   └── rules/                  
├── scripts/                    # CLI execution & evaluation runners
│   ├── cli.py
│   └── evaluate.py
├── src/
│   └── compliance_agent/       # Core application logic
│       ├── agent.py            # LangChain Agent orchestration
│       ├── models/             # Pydantic schemas
│       ├── rag/                # ChromaDB vector store & embeddings
│       └── tools/              # Code analyzer & HTML/Excel reporter
└── tests/                      # Pytest unit & integration tests
```

---

## 🧩 Extensibility

The compliance rules are entirely decoupled from the application logic. 

**Want to add a new clinical coding standard?** 
You don't need to touch a single line of Python code. Simply create a new `.json` file outlining the rule definition, evidence requirements, and severity, and drop it into the `knowledge/rules/` directory. The RAG pipeline will automatically embed and index it on the next run!

---

*Disclaimer: This is a research prototype. All compliance rules, datasets, and outputs used in this repository are synthetic and created for educational purposes. It is not affiliated with any specific pharmaceutical company or regulatory body.*
