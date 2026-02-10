# AI Code Reviewer (LLM + Heuristic Fallback)

An end-to-end **AI-powered Python code review system** built with **FastAPI**, **LangChain**, and **Hugging Face LLMs**, featuring a **deterministic heuristic fallback** for reliability and a simple **web UI** for interactive testing.

This project is designed to be **robust, modular, and extensible**, with a clean backend architecture that supports future static analysis and advanced reasoning.

---

## Features

- 🧠 **LLM-based Python code review**
- 🛡️ **Heuristic fallback** when LLM is unavailable
- 🧱 **Clean, modular backend architecture**
- 🌐 **FastAPI REST API**
- 🖥️ **Web-based UI** for testing reviews
- 🔁 **Chunk-based processing** for large code inputs
- ✅ Always returns a valid review (never crashes)

---

## Project Architecture

```
Rate_code/
│
├── backend/
│   └── app/
│       ├── core/
│       │   ├── reviewer.py
│       │   ├── chunker.py
│       │   └── aggregator.py
│       │
│       ├── models/
│       │   ├── llm.py
│       │   └── heuristic.py
│       │
│       ├── routes/
│       │   └── review.py
│       │
│       ├── schemas/
│       │   └── review.py
│       │
│       ├── utils/
│       │   └── logging.py
│       │
│       ├── config.py
│       └── main.py
│
├── web_ui.html
├── .env
└── README.md
```

---

## How the System Works

1. User pastes Python code in the web UI
2. Code is sent to the `/review` API endpoint
3. Backend splits the code into chunks
4. Each chunk is reviewed using:
   - **LLM**, if available
   - **Heuristic fallback**, if not
5. Results are aggregated into a single review
6. Final structured JSON response is returned to the UI

---

## Tech Stack

- Python 3.10+
- FastAPI
- LangChain
- Hugging Face Inference API
- Qwen2.5-Coder-32B-Instruct
- HTML + JavaScript (Tailwind CSS)

---

## Setup Instructions

### Clone the repository

```bash
git clone https://github.com/<your-username>/Rate_code.git
cd Rate_code
```

### Create environment (recommended)

```bash
conda create -n rate_code python=3.10 -y
conda activate rate_code
```

### Install dependencies

```bash
pip install fastapi uvicorn langchain langchain-huggingface python-dotenv tf-keras
```

### Configure environment variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Running the Project

```bash
uvicorn backend.app.main:app --reload
```

Open browser:

```
http://127.0.0.1:8000
```

---

## Example Output

```json
{
  "score": 8,
  "summary": "Aggregated review across code sections.",
  "suggestions": [
    "Add type hints for better readability.",
    "Add a docstring.",
    "Handle edge cases.",
    "Add unit tests.",
    "Consider performance improvements."
  ],
  "issues": []
}
```

---

## Status

✅ Step 1 completed – Clean architecture + working LLM reviewer  
🚧 Step 2 planned – AST-based static analysis

---

## License

Educational / portfolio use.
