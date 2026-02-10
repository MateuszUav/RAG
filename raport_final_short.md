
---

# Short Research Report — RAG Chatbot Migration (Bielik → Local Ollama on macOS M4 Pro)

## 1) Project Goal

The original project (`03_rag_bielik_11b_v2_6_instruct_bnb_4bit.py`) implemented a **RAG chatbot** using:

* **Bielik-11B-v2.6-Instruct** (HuggingFace + `bitsandbytes`)
* embeddings via **SentenceTransformer**
* retrieval via **FAISS**
* a custom `ask_bot()` combining retrieval + generation

Because the environment is **macOS Apple Silicon (M4 Pro)**, the original approach is unstable: **bitsandbytes/CUDA quantization is not reliable on Mac**, leading to failures and incompatibilities. Therefore the chatbot was migrated to:

✅ `ollama_rag.py`

This preserves the **RAG architecture**, but replaces the generation layer with **local Ollama**, enabling stable local inference and easier model switching. 

---

## 2) Main Modifications (What changed and why)

### 2.1 Replace HF/Bitsandbytes model with Ollama

**Before:** Bielik loaded in Python via HF + BitsAndBytes
**Now:** Ollama used through HTTP (`requests`):

* `ollama_generate(...)` for completions
* `ollama_embed(...)` for embeddings

**Reason:** stable local generation on macOS without quantization/toolchain problems.

---

### 2.2 Replace `ask_bot()` with modular Ollama pipeline

Old `ask_bot()` handled everything (retrieve → prompt → generate).
New flow splits responsibilities into clear functions:

1. `retrieve(...)` (FAISS search)
2. `build_prompt_*` (prompt construction)
3. `ollama_generate(...)` (Ollama inference)
4. output answer

Result: same logic, cleaner structure, and inference handled externally as a service.

---

### 2.3 Prompt system upgrade

Added multiple prompt builders:

* `build_prompt_classic(...)`
* `build_prompt_strict(...)`
* `build_prompt_style(...)`

This enables:

* strict grounding (answer only from context)
* refusal of irrelevant questions
* style control (tone without breaking RAG grounding)

---

## 3) AI Model Used and Why

Model selected in Ollama:

✅ **Dolphin** (uncensored/unblocked variant)

Why used:

* more compliant with style/tone prompts (slang, rude tone, etc.)
* useful for demonstrations (prompt control, refusal, injection resistance)
* runs well locally on macOS

Even though Dolphin is “unblocked”, **the RAG policy is enforced by prompt design**: strict mode forces context-only answers and refusal otherwise. 

---

## 4) How `ask_bot()` was replaced

Instead of in-process transformers generation, the model is now treated as a service:

* `ollama_generate(prompt, model=...)`
* request to `POST http://localhost:11434/api/generate`

Benefits:

* Python handles RAG logic only
* Ollama handles inference
* improved portability and stability

---

## 5) What RAG Architecture Is

**RAG (Retrieval-Augmented Generation)** combines:

1. **Retrieval:** find relevant document chunks using embeddings + FAISS
2. **Generation:** inject retrieved context into the prompt and answer using it

Purpose: prevent hallucination and ground answers in the document, so the chatbot answers based on evidence rather than general training data. 

---

## 6) Retrieval Pipeline (How context is retrieved)

In the Ollama version retrieval uses:

* `load_documents(...)`
* `chunk_text(...)` (chunking with overlap)
* `ollama_embed(...)` (vector creation)
* `build_faiss_index(...)`
* `retrieve(...)` (top-k similarity search)

Process:

1. documents loaded
2. text split into overlapping chunks
3. chunks embedded to vectors
4. stored in FAISS
5. question embedded
6. FAISS returns closest chunks + scores

The best chunks are injected into the prompt as grounding context.

---

## 7) Role of `TOP_K` and similarity score threshold

### `TOP_K`

Defines how many chunks are retrieved:

* low K → risk missing required evidence
* medium (3–5) → good balance
* high K → more evidence but more noise

### Similarity score / threshold (`MIN_SCORE`)

Controls whether retrieved chunks are “relevant enough”:

* low threshold → fewer refusals but more irrelevant context risk
* high threshold → stronger hallucination prevention but can reject valid queries

These parameters are critical for controlling when the bot answers vs refuses.

---

## 8) Embedder Function (What embeddings are for)

The embedder converts text into vectors:

* chunk vectors for indexing
* query vector for retrieval

Embeddings enable **semantic similarity**, meaning retrieval works even when wording differs. Therefore the embedder is the key component that powers RAG retrieval. 

---

## 9) Experiments and Key Findings

### RAG ON vs RAG OFF (grounding test)

* Same settings used (embed model, chat model, chunk settings, threshold, etc.)
* For **relevant question** (“what are gun range rules”), retrieval returned chunks from `shooting_range_rules.txt` and answers were grounded.
* For **unrelated question** (“kto jest prezydentem venezueli”), the model refused (“I do not know homie.”)
  This shows strict prompting prevents hallucinated world-knowledge answers.

---

### Temperature tests

* **Temp=10.0** → creative, aggressive, uncontrolled phrasing; hallucination risk increases
* **Temp=0.0** → stable, deterministic, rule-focused answer
  Conclusion: **low temperature increases reliability**, high temperature increases creativity and risk.

---

### TOP_K tests

* `TOP_K=1` retrieved wrong file only (`demo.txt`) → answer degraded/refused
* `TOP_K=10` retrieved correct rule chunks → answer detailed and accurate
  Conclusion: low K causes context loss; higher K increases evidence but may introduce noise.

---

### MIN_SCORE tests

* `MIN_SCORE=0.1` → grounded answers
* `MIN_SCORE=3.0` → no context retrieved → refuses even valid questions
* extremely low threshold accepts almost everything (risk of garbage context)
  Conclusion: thresholds trade off between hallucination prevention vs over-refusal.

---

# Conclusion

The migration from Bielik/HuggingFace to `ollama_rag.py` achieved:

* ✅ stable execution on macOS M4 Pro
* ✅ clean separation: **RAG in Python**, **LLM inference in Ollama**
* ✅ improved prompt control (strict answers, refusal, style modes)
* ✅ better experimentation (temperature, TOP_K, similarity threshold)
* ✅ ability to use Dolphin for advanced demonstration behavior

Overall, the new solution preserves the original RAG design but is **more stable, modular, and easier to experiment with on Mac hardware**. 

---

