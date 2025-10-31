🧠 ResearchBot (AutoRAG++) — Visual Research Agent

A Streamlit-based research intelligence app that automates literature review and synthesis using Retrieval-Augmented Generation (RAG) and visual explanation.
It fetches papers directly from arXiv (or user-uploaded PDFs), performs semantic search using SentenceTransformers + FAISS, summarizes insights with Ollama/Gemini, and generates flowcharts or concept graphs for visual understanding.

🚀 No external API keys required for Ollama (local Llama 3). Fully GPU-compatible.

⚙️ Key Features

📚 Smart Paper Retrieval: Search arXiv or upload PDFs directly.

🧩 RAG Pipeline: Sentence-transformer embeddings + FAISS vector index for semantic similarity search.

🧠 AI Summarization: Summarize papers, compare findings, and synthesize key insights using Ollama (Llama 3) or Gemini.

📈 Visual Research Graphs: Generate flowcharts or concept graphs to represent paper methods or core ideas.

🗂 Parallel Ingestion: Multi-threaded extraction of abstracts, pages, or full PDFs for faster indexing.

🔄 Offline-first Design: All PDF processing and embedding run locally — secure, private, and reproducible.

💾 Cache System: Stores processed PDFs and reports for reuse.

🧰 Tech Stack
Category	Technology
Frontend	Streamlit
Backend	Python 3.11
Search / Index	FAISS
Embeddings	SentenceTransformers (MiniLM / GPU)
LLM Backends	Ollama (Llama 3) / Gemini
PDF Parsing	PyMuPDF (fitz)
Visualization	NetworkX, Graphviz, Matplotlib
Async / Parallel	ThreadPoolExecutor
Data Source	arXiv API
🗂️ Project Structure
ResearchBot/
├─ streamlit_app.py
├─ requirements.txt
├─ LICENSE
├─ README.md
├─ cache/
│   ├─ pdfs/
│   ├─ texts/
├─ reports/
└─ .streamlit/
    └─ secrets.toml

💻 Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/rudryyy/ResearchBot-AutoRAG.git
cd ResearchBot-AutoRAG

2️⃣ Create a Virtual Environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ (Optional) Install GPU Version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

5️⃣ Run the App
streamlit run streamlit_app.py


Then open your browser at http://localhost:8501

🔍 Usage Workflow

Search arXiv Papers

Enter a topic (e.g., "graph neural networks for drug discovery")

Choose how much text to ingest (abstract, first N pages, or full text)

Add Custom PDFs

Upload your own research papers or paste direct PDF URLs

Embedding + Retrieval

AutoRAG extracts and embeds text chunks using MiniLM / GPU SentenceTransformer

FAISS indexes the embeddings for similarity search

Summarization

Choose an LLM backend:

Ollama (local Llama 3) → runs offline

Gemini API → for cloud summarization

ResearchBot summarizes each paper and synthesizes all findings into one final report

Visualization

View each paper’s visual explanation:

Flowchart (via LLM JSON or heuristic extraction)

Concept Graph (NetworkX-based)

Export Report

Download a consolidated Markdown report summarizing all findings

📊 Example Output
Mode	Example
Flowchart (LLM)	Extracts step-by-step research flow for a method
Concept Graph	Builds semantic networks of key technical terms
Markdown Report	Synthesized literature summary ready for presentations
🧩 Environment Variables

You can configure API keys and models in .streamlit/secrets.toml:

GEMINI_API_KEY = "your_google_genai_key"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"


Note: Ollama must be running locally if used (ollama run llama3).

🧠 Future Enhancements

Add support for PDF tables and figures extraction

Integrate HuggingFace embedding backends

Introduce auto-citation generation

Deploy as Streamlit Cloud App with persistent caching