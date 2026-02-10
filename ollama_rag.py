#dependencies
# pip install -U faiss-cpu numpy requests


import os
import re
import json
import time
import glob
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import faiss
import requests
from typer import style



# ============================================================
# CONFIG
# ============================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding model (Ollama)
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")

# Chat/generation model (Ollama)
#CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

#CHAT_MODEL = os.getenv("CHAT_MODEL", "hf.co/bartowski/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-GGUF:Q4_K_M")
CHAT_MODEL = os.getenv("CHAT_MODEL", "dolphin3:8b")

# Where your documents are
DOCS_DIR = os.getenv("DOCS_DIR", "rag_docs")

# Chunking params
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))          # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))    # characters

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))

# Debug
DEBUG = os.getenv("DEBUG", "1") == "1"

# Response style mode
STYLE_MODE = os.getenv("STYLE_MODE", "neutral")
# allowed: neutral | joke | poem | rude


# Retrieval knobs
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))         # cosine similarity (after L2 normalize)

# Generation knob
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))


# ============================================================
# DATA TYPES
# ============================================================
@dataclass
class Chunk:
    id: str
    source: str
    text: str


# ============================================================
# OLLAMA CLIENT
# ============================================================
def ollama_embed(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    """
    Uses Ollama /api/embed to embed one or multiple texts.
    Returns: np.ndarray of shape (N, D)
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {"model": model, "input": texts}

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    # Ollama returns either "embeddings" (list of vectors) or sometimes "embedding"
    if "embeddings" in data:
        emb = data["embeddings"]
    elif "embedding" in data:
        emb = [data["embedding"]]
    else:
        raise ValueError(f"Unexpected embed response keys: {list(data.keys())}")

    arr = np.array(emb, dtype=np.float32)
    return arr


def ollama_generate(prompt: str, model: str = CHAT_MODEL) -> str:
    """
    Uses Ollama /api/generate (non-stream) to generate.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
        }
    }

    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# ============================================================
# DOC LOADING + CHUNKING
# ============================================================
def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple char-based chunking with overlap.
    Works well enough to mirror your Bielik script behavior.
    """
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def load_documents(docs_dir: str = DOCS_DIR) -> List[Chunk]:
    exts = ("*.txt", "*.md")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(docs_dir, ext)))

    all_chunks: List[Chunk] = []

    for path in sorted(files):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        parts = chunk_text(raw)
        base = os.path.basename(path)
        h = file_hash(path)

        for i, p in enumerate(parts):
            cid = f"{base}:{h}:{i}"
            all_chunks.append(Chunk(id=cid, source=base, text=p))

    return all_chunks


# ============================================================
# FAISS INDEX (Cosine similarity)
# ============================================================
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norm, eps)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    For cosine similarity: normalize embeddings then use Inner Product.
    """
    embeddings = l2_normalize(embeddings.astype(np.float32))
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve(index: faiss.Index, chunks: List[Chunk], query: str, k: int = TOP_K) -> List[Tuple[Chunk, float]]:
    q_emb = ollama_embed([query])
    q_emb = l2_normalize(q_emb.astype(np.float32))

    scores, ids = index.search(q_emb, k)
    scores = scores[0]
    ids = ids[0]

    # Added MIN_SCORE filtering
    results = []
    for idx, score in zip(ids, scores):
        if idx < 0 or idx >= len(chunks):
            continue
        if float(score) < MIN_SCORE:
            continue
        results.append((chunks[idx], float(score)))

    return results



  


# ============================================================
# PROMPTING (RAG)
# ============================================================
def build_prompt_classic(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Similar to your Bielik script: CONTEXT + QUESTION + ANSWER
    """
    if retrieved:
        context_lines = []
        for ch, score in retrieved:
            context_lines.append(f"[{ch.source} | score={score:.3f}]\n{ch.text}")
        context = "\n\n---\n\n".join(context_lines)
    else:
        context = ""

    prompt = f"""You are a helpful assistant. Answer in Polish.
Use ONLY the context when it is relevant. If the context does not contain the answer, say you don't know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return prompt

# Strict RAG prompt
def build_prompt_strict(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Strict RAG prompt:
    - short answer
    - ONLY from context
    - refuse if not in context
    """
    if retrieved:
        context = "\n\n---\n\n".join(
            [f"[{ch.source} | score={score:.3f}]\n{ch.text}" for ch, score in retrieved]
        )
    else:
        context = ""

    prompt = f"""You are a QA assistant. Answer ONLY using the context below.
Rules:
- Output must be short (max 4 sentences).
- If the answer is not explicitly present in the context, respond exactly: "Nie wiem na podstawie dokumentu."
- Do not use any external knowledge.
- Do not guess.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return prompt



def style_instruction(style_mode: str) -> str:
    if style_mode == "joke":
        return "Write the answer jokingly, but still informative."
    if style_mode == "poem":
        return "Write the answer as a short rhyming poem (8-10 lines)."
    if style_mode == "rude":
        return "Write the answer using mild profanity and informal tone, be hatefull, arogant and mean. Sometimes do not answer at all. Use slurs and explicit words often"
    return "Write the answer in a neutral, professional tone."




def build_prompt_style(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:

    if retrieved:
        context = "\n\n---\n\n".join(
            [f"[{ch.source} | score={score:.3f}]\n{ch.text}" for ch, score in retrieved]
        )
    else:
        context = ""

    style = style_instruction(STYLE_MODE)

    prompt = f"""You are a QA assistant. Answer ONLY using the context below.
    Rules:
    - Output must be short (max 4 sentences) unless stated otherwise.
    - If the answer is not present in the context, respond exactly: "I do not know homie."
    - Do not use any external knowledge.
    - Do not guess.
    - Follow the STYLE_MODE instruction: 
    - {style}

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    return prompt



# ============================================================
# MAIN
# ============================================================
def main():
    print("== Loading documents ==")
    chunks = load_documents(DOCS_DIR)
    print(f"Loaded {len(chunks)} chunks from: {DOCS_DIR}")

    if len(chunks) == 0:
        print("No documents found. Put .txt/.md into DOCS_DIR and rerun.")
        return

    # Embed corpus
    print("\n== Embedding chunks with Ollama ==")
    texts = [c.text for c in chunks]

    t0 = time.time()
    embs = ollama_embed(texts)
    dt = time.time() - t0
    print(f"Embeddings shape: {embs.shape} (computed in {dt:.1f}s)")

    # Build FAISS
    print("\n== Building FAISS index ==")
    index = build_faiss_index(embs)
    print(f"FAISS total vectors: {index.ntotal}")

    print("=== Ollama RAG  ===")
    print(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
    print(f"EMBED_MODEL:     {EMBED_MODEL}")
    print(f"CHAT_MODEL:      {CHAT_MODEL}")
    print(f"DOCS_DIR:        {DOCS_DIR}")
    print(f"CHUNK_SIZE:      {CHUNK_SIZE} | CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    print(f"STYLE_MODE:      {STYLE_MODE}")
    print(f"DEBUG:           {DEBUG}")
    print(f"TOP_K:           {TOP_K} | MIN_SCORE: {MIN_SCORE}") 
    print(f"Temp:            {TEMPERATURE}")
    print()

    print("\n== Ready. Type a question (or Ctrl+C). ==")

    while True:
        try:
            q = input("\n> ")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not q.strip():
            continue

        # RAG ON
        retrieved = retrieve(index, chunks, q, TOP_K)

        if DEBUG:
            print("\n--- Retrieved context (RAG ON) ---")
            for ch, score in retrieved:
                preview = ch.text[:180].replace("\n", " ")
                print(f"- {ch.source} score={score:.3f} | {preview}...")

        prompt_on = build_prompt_style(q, retrieved)
        ans_on = ollama_generate(prompt_on)

        # RAG OFF baseline
        prompt_off = build_prompt_style(q, [])
        ans_off = ollama_generate(prompt_off)

        print("\n==============================")
        print("RAG ON answer:")
        print(ans_on)
        print("\n------------------------------")
        print("RAG OFF answer:")
        print(ans_off)
        print("==============================\n")


if __name__ == "__main__":
    main()
