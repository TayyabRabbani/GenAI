# AI Code Review Helper

This project is a lightweight code-review backend + web UI that uses a Hugging Face model to evaluate code quality.  
You send a code snippet, and the service returns a structured JSON review containing a score, summary, issues, and improvement suggestions.  
It can be used directly in the browser (`web_ui.html`) or as the backend for an editor/IDE extension.

---

## 🚀 Features

- Reviews code using a Hugging Face LLM  
  **Default model:** `Qwen/Qwen2.5-Coder-32B-Instruct`
- Automatic fallback heuristic reviewer when no API token is available
- Returns strictly formatted JSON containing:
  - **score** (0–10)
  - **summary** (short explanation)
  - **issues** (concrete problems found)
  - **suggestions** (exactly 5 actionable improvements)
- Clean TailwindCSS web interface (`web_ui.html`)
- FastAPI backend providing a single `/review` endpoint  
  → Easy to integrate into extensions, CLI tools, or automation pipelines

---

## 📁 Project Structure

```
Rate_code/
├─ .env               # Hugging Face API token (you create this)
├─ api.py             # FastAPI backend, serves /review + UI
├─ chat_model.py      # Helper to build ChatHuggingFace model
├─ code.py            # Standalone CLI-style rater + aggregator
├─ llm_model.py       # LangChain prompt + structured JSON parser
└─ web_ui.html        # Browser UI
```

---

## 🧰 Prerequisites

- **Python 3.8+**
- **Hugging Face API Token**

You can generate a token from your Hugging Face account settings and give it **read access** to models.

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>/Rate_code
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Linux / macOS**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file inside **Rate_code/**:

```
HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
```

### Ensure your code reads the token

`chat_model.py` already loads the token from environment variables.

**Important:**

- Remove any hard-coded API tokens from source files  
  (e.g., `os.environ["HUGGINGFACEHUB_API_TOKEN"] = "..."`)
- Always rely on the `.env` file.

If using `python-dotenv`, load it early inside `api.py`.

---

## ▶️ Running the Server

Inside the `Rate_code` directory:

```bash
python api.py
```

You should see:

```
Starting FastAPI server on http://localhost:5000 ...
```

### Access:

- **API Docs:** http://localhost:5000/docs  
- **Web UI:** http://localhost:5000/

---

## 🌐 Using the Web UI

1. Start the server (`python api.py`)
2. Open your browser → **http://localhost:5000/**
3. Paste your code into the text area
4. Click **Review Code**

The UI will display:

- **Overall score**
- **Short summary**
- **List of issues**
- **Five actionable suggestions**
