#!/usr/bin/env python3
# streamlit_app.py
# AutoRAG++ — Visual Research Agent (GPU SentenceTransformers • arXiv + Uploads • Ollama/Gemini • Visualizations)

import os, re, time, json, hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import arxiv
import fitz  # PyMuPDF
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Visuals
import networkx as nx
import matplotlib.pyplot as plt
try:
    from graphviz import Digraph
    GRAPHVIZ_OK = True
except Exception:
    GRAPHVIZ_OK = False

# ==============================
# CONFIG / FOLDERS
# ==============================
CACHE_DIR = Path("cache")
PDF_DIR = CACHE_DIR / "pdfs"
TXT_DIR = CACHE_DIR / "texts"
REPORTS_DIR = Path("reports")
for d in (PDF_DIR, TXT_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Embeddings
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")  # fast & light (384-dim)
DEFAULT_EMB_BATCH = 64 if torch.cuda.is_available() else 32

# LLMs
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ==============================
# HTTP SESSION (robust)
# ==============================
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "AutoRAG++/1.4 (visual research app)"})
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = make_session()

# ==============================
# UTILS
# ==============================
def safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in " .-_").rstrip()

def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

def chunk_text(text: str, chunk_size: int, overlap: int, cap: int | None = None) -> List[str]:
    text = (text or "").strip().replace("\r", "")
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks, start, L = [], 0, len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if cap and len(chunks) >= cap:
            break
        start = end - overlap
        if start < 0: start = 0
        if start >= L: break
    return chunks

# ==============================
# EMBEDDINGS (SentenceTransformers, GPU-ready)
# ==============================
@st.cache_resource(show_spinner=False)
def get_st_embedder_and_index(model_name: str):
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    return model, index

class Retriever:
    def __init__(self, model, index):
        self.model, self.index = model, index
        try: self.index.reset()
        except: pass
        self.doc_texts: List[Dict] = []

    def add_documents(self, docs: List[Dict], batch_size: int, prog_cb=None):
        payload = [(d["text"], d) for d in docs if d.get("text")]
        if not payload:
            return
        texts, kept_docs = zip(*payload)
        # Manual batching for progress
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = list(texts[i:i+batch_size])
            embs = self.model.encode(batch, batch_size=batch_size,
                                     convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(embs)
            self.index.add(embs)
            self.doc_texts.extend(list(kept_docs[i:i+batch_size]))
            if prog_cb: prog_cb(min(total, i + batch_size), total)

    def search(self, query: str, top_k=5):
        if getattr(self.index, "ntotal", 0) == 0:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        n = len(self.doc_texts)
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= n: continue
            md = self.doc_texts[idx]
            results.append({"score": float(score), "text": md["text"], "meta": md["paper_meta"], "chunk_id": md.get("chunk_id")})
        return results

# ==============================
# PDF I/O
# ==============================
def head_size_mb(url: str, timeout: int) -> float | None:
    try:
        r = SESSION.head(url, allow_redirects=True, timeout=timeout)
        cl = r.headers.get("Content-Length")
        return (int(cl) / (1024 * 1024)) if cl else None
    except Exception:
        return None

def download_pdf(pdf_url: str, arxiv_id: str, timeout: int, retries: int, max_mb: int) -> str | None:
    local = PDF_DIR / f"{safe_filename(arxiv_id)}.pdf"
    if local.exists():
        return str(local)
    size = head_size_mb(pdf_url, timeout=min(10, timeout))
    if size is not None and size > max_mb:
        return None
    last_err = None
    for _ in range(max(1, retries + 1)):
        try:
            with SESSION.get(pdf_url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(1 << 15):
                        if chunk: f.write(chunk)
            return str(local)
        except Exception as e:
            last_err = e
            time.sleep(0.7)
    raise RuntimeError(f"download failed for {arxiv_id}: {last_err}")

def cached_pdf_to_text_first_pages(pdf_path: str, max_pages: int) -> str:
    txt_path = TXT_DIR / (Path(pdf_path).stem + f".first{max_pages}.txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="ignore")
    doc = fitz.open(pdf_path)
    pieces = []
    for i, page in enumerate(doc):
        if i >= max_pages: break
        try:
            pieces.append(page.get_text("text"))
        except Exception:
            pieces.append(page.get_text())
    doc.close()
    text = "\n\n".join(pieces)
    txt_path.write_text(text, encoding="utf-8", errors="ignore")
    return text

def cached_pdf_to_text_full(pdf_path: str) -> str:
    txt_path = TXT_DIR / (Path(pdf_path).stem + ".full.txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="ignore")
    doc = fitz.open(pdf_path)
    pieces = [pg.get_text("text") for pg in doc]
    doc.close()
    text = "\n\n".join(pieces)
    txt_path.write_text(text, encoding="utf-8", errors="ignore")
    return text

def save_uploaded_pdf(file) -> str:
    local = PDF_DIR / safe_filename(file.name)
    with open(local, "wb") as f:
        f.write(file.getbuffer())
    return str(local)

def add_user_paper_from_path(local_path: str) -> Dict:
    stem = Path(local_path).stem
    arx_id = f"upload-{hash_text(local_path)[:10]}"
    return {
        "title": stem.replace("_"," "),
        "authors": ["(uploaded)"],
        "summary": "",
        "published": "",
        "pdf_url": local_path,
        "entry_id": arx_id,
        "arxiv_id": arx_id,
        "local_pdf": local_path,
    }

def add_user_paper_from_url(url: str) -> Dict:
    try:
        name = Path(url.split("?")[0]).name or f"user_{hash_text(url)[:8]}.pdf"
        local = PDF_DIR / safe_filename(name)
        if not local.exists():
            with SESSION.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(1 << 15):
                        if chunk: f.write(chunk)
        return add_user_paper_from_path(str(local))
    except Exception as e:
        st.warning(f"Could not fetch URL PDF: {e}")
        return {}

# ==============================
# arXiv SEARCH (no deprecation)
# ==============================
@st.cache_data(show_spinner=False)
def cached_search_arxiv(query: str, max_results: int):
    client = arxiv.Client(page_size=min(max_results, 50), delay_seconds=2, num_retries=2)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = list(client.results(search))
    papers = []
    for r in results:
        papers.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "summary": r.summary or "",
            "published": r.published.isoformat() if r.published else "",
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id,
            "arxiv_id": r.entry_id.split("/")[-1],
        })
    return papers

# ==============================
# LLMs (Ollama + Gemini + Fallback)
# ==============================
def ollama_chat(messages: List[dict], temperature: float = 0.0) -> str:
    try:
        payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_ctx": 4096}}
        r = SESSION.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content") or data.get("content") or ""
    except Exception as e:
        return f"[Ollama error: {e}]"

def gemini_chat(messages: List[dict], api_key: str, model_name: str, temperature: float = 0.0) -> str:
    if not api_key:
        return "[Gemini error: Missing API key]"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        prompt = "\n".join(m["content"] for m in messages)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"[Gemini error: {e}]"

def llm_chat(messages: List[dict], backend: str, gemini_key: str, model_name: str) -> str:
    if backend.startswith("Ollama"):
        return ollama_chat(messages)
    elif backend.startswith("Gemini"):
        return gemini_chat(messages, gemini_key, model_name)
    return "[LLM disabled]"

def naive_summary(text: str, max_chars: int = 1000) -> str:
    # simple fallback: take first N non-empty sentences
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for s in sents:
        if len(" ".join(out)) >= max_chars: break
        if s.strip(): out.append(s.strip())
    return " ".join(out) if out else text[:max_chars]

SYSTEM_PROMPT_SUMMARIZE = (
    "Summarize the snippets into: 1) a concise paragraph, 2) 4–6 bullets (methods/results/novelty), "
    "3) any contradictions or open questions. Use [arXiv:ID] inline when appropriate. Be factual."
)
SYSTEM_PROMPT_SYNTHESIS = (
    "Synthesize the paper-level summaries into a short research report with: title, problem, background (2–3 lines), "
    "consensus (4–6 bullets), open questions (2–4), and 3 recommended future directions. Keep concise."
)
SYSTEM_PROMPT_FLOW_JSON = """Extract a minimal step-by-step flow of the method as JSON:
{"steps":[{"id":"S1","label":"..."},{"id":"S2","label":"..."}],
 "edges":[["S1","S2"],["S2","S3"]]}
Short labels (<=8 words), 4–8 steps. Return ONLY JSON.
"""

def summarize_snippets(snippets_text: str, backend: str, gemini_key: str) -> str:
    if backend == "None (no LLM)":
        return "Summary (extract-only): " + naive_summary(snippets_text)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": snippets_text}]
    out = llm_chat(msgs, backend, gemini_key, GEMINI_MODEL)
    if out.startswith("[Ollama error") or out.startswith("[Gemini error"):
        return "Summary (fallback): " + naive_summary(snippets_text)
    return out

def synthesize_report(combined_summaries: str, backend: str, gemini_key: str) -> str:
    if backend == "None (no LLM)":
        return "Synthesis (extract-only):\n\n" + combined_summaries[:2000]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_SYNTHESIS},
            {"role": "user", "content": combined_summaries}]
    out = llm_chat(msgs, backend, gemini_key, GEMINI_MODEL)
    if out.startswith("[Ollama error") or out.startswith("[Gemini error"):
        return "Synthesis (fallback):\n\n" + combined_summaries[:2000]
    return out

def llm_flow_json_from_text(title: str, snippets_text: str, backend: str, gemini_key: str) -> dict | None:
    if backend == "None (no LLM)":
        return None
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_FLOW_JSON},
            {"role": "user", "content": f"Paper: {title}\nSnippets:\n{snippets_text}\n\nReturn JSON only."}]
    out = llm_chat(msgs, backend, gemini_key, GEMINI_MODEL)
    try:
        start = out.find('{'); end = out.rfind('}')
        if start >= 0 and end > start:
            return json.loads(out[start:end+1])
        return json.loads(out)
    except Exception:
        return None

# ==============================
# VISUALS
# ==============================
STOPWORDS = set("""
a an the of for with to and or in on from by over under between into across as at via
is are was were be been being this that these those it they them we you he she their our
using based approach method model network graph data results experiments baseline proposed
""".split())

def extract_keyterms(text: str, top_k: int = 20) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) >= 4]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_k)]

def build_cooccurrence_graph(text: str, window: int = 12, top_k: int = 20) -> nx.Graph:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 4]
    keyset = set(extract_keyterms(text, top_k=top_k))
    G = nx.Graph()
    for k in keyset: G.add_node(k)
    for i in range(len(tokens)):
        if tokens[i] not in keyset: continue
        for j in range(i+1, min(i+window, len(tokens))):
            if tokens[j] not in keyset or tokens[i] == tokens[j]: continue
            w = G.get_edge_data(tokens[i], tokens[j], {}).get("weight", 0) + 1
            G.add_edge(tokens[i], tokens[j], weight=w)
    return G

def render_concept_graph(text: str, max_nodes: int = 20):
    G = build_cooccurrence_graph(text, window=10, top_k=max_nodes)
    if len(G) == 0:
        st.info("Not enough signal to build a concept graph.")
        return
    if not nx.is_empty(G):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if comps: G = G.subgraph(comps[0]).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    deg = dict(G.degree())
    sizes = [300 + 60*deg[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

def render_flowchart_from_json(flow: dict):
    if not flow or not flow.get("steps"):
        st.info("No flow steps to visualize.")
        return
    if GRAPHVIZ_OK:
        dot = Digraph(format="png")
        for s in flow.get("steps", []):
            dot.node(s.get("id","?"), s.get("label",""))
        for u, v in flow.get("edges", []):
            dot.edge(u, v)
        st.graphviz_chart(dot)
    else:
        steps = [s.get("label","") for s in flow.get("steps", [])]
        render_flowchart_matplotlib(steps)

def render_flowchart_matplotlib(steps: List[str]):
    if not steps:
        st.info("No steps to visualize.")
        return
    fig, ax = plt.subplots(figsize=(7, max(4, 0.8*len(steps))))
    ax.axis("off")
    box_w, box_h = 0.8, 0.12
    y = 0.9
    coords = []
    for s in steps:
        ax.add_patch(plt.Rectangle((0.1, y - box_h/2), box_w, box_h, fill=False, linewidth=1.5))
        lbl = re.sub(r"\s+", " ", s)[:70] + ("…" if len(s) > 70 else "")
        ax.text(0.1 + box_w/2, y, lbl, ha="center", va="center", fontsize=9)
        coords.append((0.1 + box_w/2, y - box_h/2))
        y -= 0.18
    for i in range(len(coords)-1):
        x0, y0 = coords[i]; x1, y1 = coords[i+1]
        ax.annotate("", xy=(x1, y1 + box_h + 0.02), xycoords="axes fraction",
                    xytext=(x0, y0 - 0.02), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.2))
    st.pyplot(fig, clear_figure=True)

def render_flowchart_heuristic(text: str):
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    verbs = ("propos", "present", "introduc", "collect", "preprocess", "train", "optim",
             "evaluat", "compar", "report", "conclud", "deploy")
    key = []
    for s in sents:
        s_l = s.lower()
        if any(v in s_l for v in verbs) and 10 < len(s) < 220:
            key.append(s.strip())
        if len(key) >= 8: break
    if not key:
        key = [s.strip() for s in sents if len(s.strip()) > 0][:5]
    render_flowchart_matplotlib(key)

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="AutoRAG++ (Visual Research Agent)", layout="wide")
st.title("🔎 AutoRAG++ — Visual Research Agent (GPU embeddings • Ollama/Gemini)")

with st.sidebar:
    st.subheader("Papers")
    query = st.text_input("arXiv query", value="graph neural networks for drug discovery")
    num_papers = st.slider("Max arXiv papers", 1, 25, 8, 1)
    use_user_only = st.toggle("Only use my uploads", value=False)
    user_pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    user_pdf_url = st.text_input("Or paste a direct PDF URL")

    st.subheader("Ingestion")
    ingestion_mode = st.radio("How much text?", ["Abstracts only (fastest)", "Abstract + first N pages", "Full text"])
    max_pages = st.slider("If 'first N pages'", 2, 30, 8, 1)
    chunk_size = st.slider("Chunk size (chars)", 800, 4000, 2000, 100)
    chunk_overlap = st.slider("Overlap (chars)", 0, 800, 100, 25)
    max_chunks_per_paper = st.slider("Max chunks per paper (cap)", 10, 200, 80, 10)

    st.subheader("Performance / Limits")
    max_pdf_mb = st.slider("Skip PDFs larger than (MB)", 5, 100, 25, 5)
    pdf_timeout = st.slider("Per-PDF timeout (sec)", 10, 180, 45, 5)
    retries = st.slider("Download retries", 0, 3, 1)
    max_workers = st.slider("Parallel workers (I/O)", 1, 12, 6, 1)
    emb_batch = st.slider("Embedding batch size", 8, 256, DEFAULT_EMB_BATCH, 8)
    st.caption(f"Embeddings: {EMB_MODEL} • device: {'cuda' if torch.cuda.is_available() else 'cpu'} • batch={emb_batch}")

    st.subheader("RAG & LLM")
    top_k_papers = st.slider("Papers to summarize", 1, 10, 5, 1)
    top_k_passages = st.slider("Passages per paper", 3, 30, 8, 1)
    backend = st.selectbox("LLM backend", ["None (no LLM)", "Ollama (Llama 3)", "Gemini"])
    gemini_key = st.text_input("Gemini API Key (optional if using Gemini)", type="password", value=os.getenv("GEMINI_API_KEY",""))

    st.subheader("Visual explanations")
    visual_mode = st.toggle("Explain papers visually", value=True)
    visual_type = st.selectbox("Visual style", ["Flowchart (LLM JSON / heuristic)", "Concept Graph (NetworkX)"])
    visual_nodes = st.slider("Max concepts / steps", 5, 25, 12, 1)
    if visual_type.startswith("Flowchart") and not GRAPHVIZ_OK:
        st.caption("Tip: Install Graphviz for nicer flowcharts: pip install graphviz & system Graphviz.")

    run_btn = st.button("Run Research", type="primary", use_container_width=True)

tabs = st.tabs(["Overview", "Per-paper Summaries (with Visuals)", "Final Report", "Logs"])
ov_tab, sum_tab, report_tab, log_tab = tabs
log_box = log_tab.container()
def log(msg: str): log_box.write(msg)

if run_btn:
    # 1) Collect papers
    papers: List[Dict] = []
    if not use_user_only and query.strip():
        with st.status("Searching arXiv…", expanded=False) as s1:
            try:
                papers = cached_search_arxiv(query, max_results=num_papers)
                s1.update(label=f"Found {len(papers)} arXiv papers", state="complete")
            except Exception as e:
                s1.update(label=f"arXiv search error: {e}", state="error")

    user_list: List[Dict] = []
    if user_pdfs:
        for f in user_pdfs:
            pth = save_uploaded_pdf(f)
            user_list.append(add_user_paper_from_path(pth))
    if user_pdf_url.strip():
        maybe = add_user_paper_from_url(user_pdf_url.strip())
        if maybe: user_list.append(maybe)

    papers = (user_list if use_user_only else (user_list + papers))
    if not papers:
        st.error("No papers to process. Upload a PDF or run a search.")
        st.stop()

    with ov_tab:
        st.subheader("Papers to process")
        for p in papers:
            source = "Uploaded" if p["arxiv_id"].startswith("upload-") else "arXiv"
            with st.expander(f"{p['title']}  [{p['arxiv_id']}]"):
                st.write(f"**Source:** {source}")
                st.write(f"**Authors:** {', '.join(p['authors'][:10]) if p.get('authors') else '(unknown)'}")
                st.write(f"**Published:** {p.get('published','')}")
                st.write(f"**PDF:** {p.get('pdf_url','')}")
                if p.get("summary"): st.write(p["summary"])

    # 2) Embedder + FAISS
    model, index = get_st_embedder_and_index(EMB_MODEL)
    retriever = Retriever(model, index)

    # 3) Ingest text (parallel fetch/extract)
    prog_fetch = st.progress(0.0, text="Preparing papers…")
    total = len(papers)

    def process_one(p: Dict) -> Tuple[str, str, str]:
        arx_id = p["arxiv_id"]
        try:
            if ingestion_mode == "Abstracts only (fastest)":
                return arx_id, p.get("summary", ""), "abstract"
            local_path = p.get("local_pdf")
            if not local_path and p.get("pdf_url") and Path(str(p["pdf_url"])).exists():
                local_path = p["pdf_url"]
            if ingestion_mode == "Abstract + first N pages":
                if not local_path:
                    local_path = download_pdf(p["pdf_url"], arx_id, timeout=pdf_timeout, retries=retries, max_mb=max_pdf_mb)
                    if local_path is None:
                        return arx_id, p.get("summary",""), f"skipped >{max_pdf_mb}MB → abstract"
                text = cached_pdf_to_text_first_pages(local_path, max_pages=max_pages)
                return arx_id, text if text.strip() else p.get("summary",""), f"first {max_pages} pages"
            # Full text
            if not local_path:
                local_path = download_pdf(p["pdf_url"], arx_id, timeout=pdf_timeout, retries=retries, max_mb=max_pdf_mb)
                if local_path is None:
                    return arx_id, p.get("summary",""), f"skipped >{max_pdf_mb}MB → abstract"
            text = cached_pdf_to_text_full(local_path)
            return arx_id, text if text.strip() else p.get("summary",""), "full text"
        except Exception as e:
            return arx_id, p.get("summary",""), f"error → abstract ({e})"

    all_docs: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, p) for p in papers]
        finished = 0
        for fut in as_completed(futures):
            arx_id, text, note = fut.result()
            finished += 1
            prog_fetch.progress(finished / max(1, total))
            p = next(pp for pp in papers if pp["arxiv_id"] == arx_id)
            if note: log(f"ℹ️ {arx_id}: {note}")
            # chunk & cap
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap, cap=max_chunks_per_paper)
            for j, c in enumerate(chunks):
                all_docs.append({
                    "text": c,
                    "paper_meta": {
                        "title": p["title"],
                        "authors": p.get("authors", []),
                        "arxiv_id": p["arxiv_id"],
                        "pdf_url": p.get("pdf_url",""),
                        "published": p.get("published",""),
                    },
                    "chunk_id": f"{p['arxiv_id']}_chunk_{j}",
                })

    # 4) Embeddings + index (with progress)
    total_chunks = len(all_docs)
    if total_chunks == 0:
        st.error("No text extracted to embed. Try 'Abstract + first N pages' or upload a different PDF.")
        st.stop()

    prog_emb = st.progress(0.0, text=f"Embedding {total_chunks} chunks…")
    last = {"done": 0}
    def emb_prog(done, total):
        last["done"] = done
        prog_emb.progress(done/total, text=f"Embedding {done}/{total} chunks…")

    retriever.add_documents(all_docs, batch_size=emb_batch, prog_cb=emb_prog)
    prog_emb.progress(1.0, text=f"Embedding complete ({last['done']}/{total_chunks})")

    # 5) Per-paper summaries (+ visuals)
    per_paper_summaries = []
    with st.status("Summarizing papers…", expanded=False) as s2:
        for p in papers[:top_k_papers]:
            q = f"Summarize findings related to '{query}' in the paper: {p['title']}"
            hits = retriever.search(q, top_k=top_k_passages)
            if not hits:
                # Fallback: search by title only
                hits = retriever.search(p["title"], top_k=top_k_passages)
            snippets_text = "\n\n".join(
                f"[{h['meta']['arxiv_id']}] {h['meta']['title']} — snippet:\n{h['text'][:800].strip()}"
                for h in hits
            )
            if not snippets_text.strip():
                snippets_text = p.get("summary","") or f"(No text extracted for {p['title']})"
            summary = summarize_snippets(snippets_text, backend, gemini_key)
            per_paper_summaries.append({"paper": p, "summary": summary, "snippets": snippets_text})
            time.sleep(0.02)
        s2.update(label="Per-paper summaries complete", state="complete")

    with tabs[1]:
        st.subheader("Per-paper Summaries (with Visuals)")
        for x in per_paper_summaries:
            p = x["paper"]
            st.markdown(f"### {p['title']}  [{p['arxiv_id']}]")
            authors = ", ".join(p.get("authors", [])[:8]) if p.get("authors") else "(unknown)"
            st.write(f"**Authors:** {authors}")
            st.write(x["summary"])
            if p.get("pdf_url"): st.caption(p["pdf_url"])
            st.divider()
            if visual_mode:
                st.markdown("**Visual explanation:**")
                if visual_type.startswith("Flowchart"):
                    flow = llm_flow_json_from_text(p["title"], x["snippets"], backend, gemini_key)
                    if flow and flow.get("steps"):
                        render_flowchart_from_json(flow)
                    else:
                        render_flowchart_heuristic(x["snippets"])
                else:
                    render_concept_graph(x["snippets"], max_nodes=visual_nodes)

    # 6) Final synthesis + download
    combined = "\n\n".join([f"[{x['paper']['arxiv_id']}] {x['paper']['title']}\n{x['summary']}" for x in per_paper_summaries])
    final_report = synthesize_report(combined, backend, gemini_key)

    with report_tab:
        st.subheader("Synthesis Report")
        st.markdown(final_report)
        qhash = hash_text(query)[:8]
        out_md = REPORTS_DIR / f"report_{safe_filename(query)[:40]}_{qhash}.md"
        with out_md.open("w", encoding="utf-8") as f:
            f.write(f"# AutoRAG++ Report for: {query}\n\n")
            f.write(f"Generated: {time.asctime()}\n\n")
            f.write("## Synthesis Report\n\n")
            f.write(final_report + "\n\n")
            f.write("## Per-paper Summaries\n\n")
            for x in per_paper_summaries:
                p = x["paper"]
                f.write(f"### {p['title']}  [{p['arxiv_id']}]\n")
                if p.get("authors"): f.write(f"Authors: {', '.join(p['authors'][:6])}\n\n")
                f.write(x["summary"] + "\n\n")
                if p.get("pdf_url"): f.write(f"PDF: {p['pdf_url']}\n\n")
                f.write("---\n\n")
        st.download_button("⬇️ Download Markdown report", data=out_md.read_bytes(),
                           file_name=out_md.name, mime="text/markdown", use_container_width=True)

    st.success("Done ✅")

else:
    with ov_tab:
        st.info("Enter a query and/or upload PDFs, then click **Run Research**. Use **Abstract + first N pages** for speed.")
