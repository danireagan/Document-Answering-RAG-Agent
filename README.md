# CV Question & Answer Agent

A Python-based agentic AI system that answers questions about a CV using LangGraph, GPT-4o, and ChromaDB. Answers are strictly grounded in CV content with no hallucinations.

## Architecture

```
User Question
      ↓
 [agent_node]          LLM with bound tools (GPT-4o)
      ↓
    tools_condition
      │
      ├── tool_calls → [tool_node]     search_cv / get_cv_sections
      │                    │
      │                    └──→ [agent_node]   LLM generates grounded answer
      │
      └── done → [output_formatter]   Appends citation block
                        ↓
                    output.txt + Console/UI
```

**Stack:** LangGraph · GPT-4o (OpenAI) · ChromaDB · HuggingFace Embeddings (all-MiniLM-L6-v2)· Streamlit

---

## Prerequisites

- Python 3.12
- OpenAI API key

---

## Installation

```bash
git clone <your-repo-url>
cd cv_agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY
```

---

## Configuration

All configuration is driven by environment variables loaded via Pydantic Settings.
Edit `.env` to customise:

| Variable                 | Default          | Description                            |
| ------------------------ | ---------------- | -------------------------------------- |
| `OPENAI_API_KEY`         | —                | OpenAI API key (required)              |
| `OPENAI_MODEL`           | `gpt-4o`         | OpenAI model name                      |
| `OPENAI_TEMPERATURE`     | `0.0`            | LLM temperature                        |
| `OPENAI_MAX_TOKENS`      | `1024`           | Max tokens per response                |
| `PDF_CHUNK_SIZE`         | `512`            | Characters per chunk for PDF files     |
| `PDF_CHUNK_OVERLAP`      | `64`             | Overlap between PDF chunks             |
| `DOCX_CHUNK_SIZE`        | `256`            | Characters per chunk for DOCX files    |
| `DOCX_CHUNK_OVERLAP`     | `32`             | Overlap between DOCX chunks            |
| `RETRIEVAL_TOP_K`        | `5`              | CV chunks retrieved per question       |
| `CV_DIR`                 | `data/cv`        | Directory for CV files                 |
| `CHROMA_PERSIST_DIR`     | `data/chroma_db` | ChromaDB storage path                  |
| `LOG_LEVEL`              | `INFO`           | Logging level                          |
| `MAX_CONVERSATION_TURNS` | `20`             | History window (20 turns × 4 messages) |

PDF and DOCX use separate chunk sizes because their structure differs: PDF pages are dense prose (benefits from larger chunks), while DOCX files are line-by-line (benefits from smaller, more granular chunks).

---

## Add Your CV

Place your CV in `data/cv/`:

```
data/cv/your_name_cv.pdf    # PDF format
data/cv/your_name_cv.docx   # or DOCX format
```

Only one CV file should be in `data/cv/` at a time.

---

## Run CLI

```bash
# Interactive mode (default)
python -m cli.main

# Force re-ingestion of CV
python -m cli.main --re-ingest

# Single question, then enter interactive mode
python -m cli.main --query "What programming languages do you know?"

# Single question, then exit
python -m cli.main --query "What is your education background?" --no-interactive
```

**CLI commands during session:**

- `clear` — reset conversation history
- `exit` / `quit` — end the session

---

## Run Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### CV Source Options

**Mode 1 — Upload file:**

1. Sidebar → CV Source → select "Upload file"
2. Click Browse and select your PDF or DOCX
3. Click "Ingest Uploaded CV"

**Mode 2 — Use existing file:**

1. Place your CV in `data/cv/`
2. Sidebar → CV Source → select "Use existing file"
3. Select from the dropdown
4. Click "Ingest Selected CV"

### Model Switcher

Switch between `gpt-4o`, `gpt-4o-mini`, and `gpt-3.5-turbo` live from the sidebar without restarting the app.

---

## Output

Every session automatically saves all Q&A to:

```
output/output.txt
```

The file is created on first run and appended on subsequent runs.
Each entry includes the question, answer, citations, and timestamp.

**Logs** are written to `logs/app.log` (JSON structured) and the console (human-readable).

---

## Project Structure

```
cv_agent/
├── config.py                    # Pydantic Settings — single source of truth
├── .env.example                 # Environment variable template
├── requirements.txt
│
├── data/cv/                     # Place your CV here (PDF or DOCX)
├── output/                      # Q&A output.txt written here at runtime
├── logs/                        # app.log written here at runtime
│
├── src/
│   ├── ingestion/               # CV loading and chunking
│   │   ├── loader.py            # Scans cv/ dir, dispatches to parsers
│   │   ├── pdf_parser.py        # pdfplumber — page-level extraction
│   │   ├── docx_parser.py       # python-docx — flat line-by-line extraction
│   │   └── chunker.py           # RecursiveCharacterTextSplitter + metadata
│   ├── vectorstore/             # Embedding and retrieval
│   │   ├── store.py             # ChromaDB PersistentClient, upsert, singleton
│   │   └── retriever.py         # similarity_search, EmptyRetrievalError
│   ├── agent/                   # LangGraph agent
│   │   ├── state.py             # AgentState TypedDict + make_initial_state()
│   │   ├── tools.py             # search_cv + get_cv_sections @tool definitions
│   │   ├── nodes.py             # agent_node + output_formatter_node
│   │   └── graph.py             # StateGraph: agent → tools → agent → formatter
│   ├── llm/
│   │   └── factory.py           # get_llm() — GPT-4o singleton
│   ├── output/
│   │   └── writer.py            # Thread-safe OutputWriter for output.txt
│   └── utils/
│       └── logging_config.py    # JSONFormatter, setup_logging(), get_logger()
│
├── cli/
│   └── main.py                  # Interactive CLI + single-shot mode
│
└── ui/
    └── streamlit_app.py         # Streamlit chat UI with sidebar controls
```

---

## Troubleshooting

| Problem                      | Solution                                                   |
| ---------------------------- | ---------------------------------------------------------- |
| `OPENAI_API_KEY must be set` | Add `OPENAI_API_KEY=sk-...` to `.env`                      |
| `No PDF or DOCX files found` | Place your CV in `data/cv/`                                |
| `Could not parse '...'`      | Check file is not corrupted or password-protected          |
| ChromaDB empty after restart | Run `python -m cli.main --re-ingest`                       |
| Streamlit port in use        | Run `streamlit run ui/streamlit_app.py --server.port=8502` |
| Slow first run               | HuggingFace model downloads on first use (~80MB)           |

If streamlit or cli run throws "can’t resolve the src package from your current run context" error, add the full path folder location to PYTHONPATH environment variable as $env:PYTHONPATH="ProjectFolderLocation" or use python install -e . command
"# Document-Answering-RAG-Agent" 
