# Local RAG with Ollama & FAISS

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that runs entirely locally. It allows you to chat with your own `.txt` or `.md` documents by combining **Ollama** for embeddings/generation and **FAISS** for high-performance vector similarity search.

---

## 🚀 Features

* **Fully Local:** Data never leaves your machine.
* **Vector Search:** Uses FAISS with L2-normalized Inner Product for fast cosine similarity.
* **Dynamic Personalities:** Supports multiple "Style Modes" (Professional, Joke, Poem, or Rude/Arrogant).
* **Smart Chunking:** Character-based chunking with configurable overlap.
* **Baseline Comparison:** Displays both "RAG ON" (using context) and "RAG OFF" (model's internal knowledge) responses simultaneously.

---

## 🛠 Prerequisites

1. **Ollama Installed:** [ollama.com](https://ollama.com)
2. **Models Downloaded:**
```bash
ollama pull embeddinggemma
ollama pull dolphin3:8b

```


3. **Python Dependencies:**
```bash
pip install -U faiss-cpu numpy requests

```



---

## ⚙️ Configuration

The script uses environment variables for easy configuration. You can change these in your terminal before running the script:

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | API endpoint for Ollama. |
| `EMBED_MODEL` | `embeddinggemma` | The model used for vector embeddings. |
| `CHAT_MODEL` | `dolphin3:8b` | The model used for text generation. |
| `DOCS_DIR` | `rag_docs` | Folder containing your `.txt` or `.md` files. |
| `STYLE_MODE` | `neutral` | Response style: `neutral`, `joke`, `poem`, or `rude`. |
| `TOP_K` | `4` | Number of document chunks to retrieve. |
| `MIN_SCORE` | `0.30` | Minimum similarity threshold for retrieval. |

---

## 📖 How to Use

1. **Prepare Documents:** Place your text files in the `rag_docs` folder.
2. **Run the Script:**
```bash
python your_script_name.py

```


3. **Interaction:**
* The script will index your files on startup.
* Type your question at the `>` prompt.
* **DEBUG Mode:** If enabled (`DEBUG=1`), it will show exactly which document chunks were found and their similarity scores.



---

## 🧠 Logic Flow

1. **Ingestion:** Reads documents, cleans whitespace, and splits them into overlapping chunks.
2. **Indexing:** Sends chunks to Ollama's `/api/embed` endpoint and stores them in a FAISS flat index.
3. **Retrieval:** When a query is asked, it is converted into a vector. FAISS finds the top  most similar chunks.
4. **Generation:** A prompt is built including the retrieved context, your question, and the selected style instructions.

---

