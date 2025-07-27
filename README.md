# PDF / URL Analyser + Chatbot (Streamlit)

A lightweight Retrieval-Augmented Generation (RAG) demo.  
Upload PDFs or paste URLs → chat with the content using **Mistral-7B-Instruct** (via OpenRouter).

---

## 📋 Step-by-Step Guide

| # | Action | Command / Link |
|---|--------|----------------|
| **1** | **Clone the repo** | `git clone https://github.com/Crudcook/vedant_projects.git && cd vedant_projects` |
| **2** | *(Optional)* create & activate venv | `python -m venv .venv && source .venv/bin/activate`  (Windows: `.venv\Scripts\activate`) |
| **3** | **Install requirements** | `pip install -r requirements.txt` |
| **4** | **Create an OpenRouter account** | <https://openrouter.ai> → Sign up / Log in |
| **5** | **Enable the model** | In **Models** search **mistralai/mistral-7b-instruct:free** → **Enable** |
| **6** | **Generate an API key** | Avatar ▷ **API Keys** ▷ **Generate new key** → copy `sk-or-…` |
| **7** | **Copy the API key in the code** |
| **8** | **Test the key & model** | ```bash
curl https://openrouter.ai/api/v1/chat/completions \
 -H "Authorization: Bearer $OPENROUTER_API_KEY" \
 -H "Content-Type: application/json" \
 -d '{"model":"mistralai/mistral-7b-instruct:free","messages":[{"role":"user","content":"Ping"}],"max_tokens":2}'
``` |
| **9** | **Run Streamlit app** | `streamlit run app.py` |
| **10** | **Use the UI** | 1 Click **Process** after adding PDFs/URLs ✓ <br>Ask questions in the chat panel ✓ |

---

## ⚙️ Configuration (in `app.py`)

| Constant | Meaning | Default |
|----------|---------|---------|
| `CHUNK_SIZE` | words per chunk | 800 |
| `SIM_THRESHOLD` | retrieval cosine gate | 0.10 |
| `RETRY_ATTEMPTS` | OpenRouter retries on 429/5xx | 4 |
| `SAMPLE_QA_N` | auto sample Q-A pairs | 1 |

Change → save → restart **Streamlit**.