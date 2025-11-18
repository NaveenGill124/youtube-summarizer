# 🏗️ Project Architecture — YouTube Summarizer (Chrome Extension + FastAPI RAG Backend)

This document explains the complete architecture of the YouTube Summarizer project, including extension flow, backend RAG pipeline, data storage, and execution logic.

---

# 📂 Folder Structure
```
youtube-summarizer/
├── backend/
│ ├── main.py # FastAPI application (RAG pipeline)
│ ├── requirements.txt # Python dependencies
│ ├── .env # Environment variables (ignored)
│ └── vectorstores/ # FAISS indexes (gitignored)
│ └── {video_id}/
│ ├── index.faiss # Vector embeddings
│ ├── chunks.json # Preprocessed text chunks
│ ├── metadata.json # Timestamp metadata
│ └── transcript.json # Raw transcript
│
├── manifest.json # Chrome extension config
├── content_script.js # In-video overlay (Shadow DOM UI)
├── popup.html # Extension popup UI
├── popup.js # Logic for popup UI
├── icons/ # All extension icons (16, 48, 128)
│
├── README.md # Main Project Documentation
└── ARCHITECTURE.md # This file


```

---

# ⚙️ System Architecture Overview

The system consists of:

### **1️⃣ Chrome Extension**
- Detects YouTube video ID
- Injects floating AI overlay using Shadow DOM
- Sends queries to backend over REST
- Displays summary, QnA, key points, timestamps

### **2️⃣ FastAPI Backend**
Implements a complete RAG pipeline:
- Extracts YouTube transcript (3 fallback methods)
- Splits transcript into chunks (time-aware)
- Generates embeddings using OpenAI
- Stores vectors in FAISS index
- Queries vectors based on user questions
- Produces answer using GPT-4o-mini

### **3️⃣ Vector Store (FAISS)**
Stores:
- FAISS index
- chunk text
- timestamp metadata
- raw transcript

---

# 🔄 Data Flow Diagram

               ┌───────────────────────────┐
               │        Chrome Extension   │
               │ ┌──────────────┐          │
               │ │ Popup UI     │          │
               │ └──────────────┘          │
               │ ┌──────────────┐          │
               │ │ Overlay UI   │──Render→ Shadow DOM
               │ └──────────────┘          │
               └─────────▲─────────────────┘
                         │  Response
                         │
                ┌────────┴───────────┐
                │    FastAPI Backend │
                │                    │
                │ Transcript Extract │← Fetch ← YouTube
                │ Chunking           │
                │ Embeddings         │→ OpenAI text-embedding-3-small
                │ Vector Store (FAISS) 
                │ RAG Query Handler  │→ GPT-4o-mini (LLM Answer)
                └─────────▲──────────┘
                          │
                          │ Store
                   ┌──────┴────┐
                   │ Vector DB │
                   └───────────┘



---

# 🔍 Backend RAG Pipeline Details

### **1. Transcript Extraction**
Backend uses 3 fallback methods:

1. YouTube timedtext API  
2. HTML caption scraping  
3. `yt-dlp` auto extractor (**99% success rate**)  

Transcript saved as:  
`vectorstores/{video_id}/transcript.json`

---

### **2. Chunking (Time-Based)**
Each transcript is split:

- Chunk length: **~120 seconds**
- Overlap: **20 seconds**
- Metadata stored with timestamps

This ensures answer accuracy and relevance.

---

### **3. Embedding Generation**
Uses:

text-embedding-3-small


Embeddings stored in FAISS index:
`index.faiss`

---

### **4. Vector Search (FAISS)**
On each user question:
- Query embedded
- Top-k vectors retrieved
- Chunk text injected into LLM prompt

---

### **5. LLM Answer Generation**
Uses:
gpt-4o-mini


LLM generates:
- Summary  
- QnA  
- Key Insights  
- Timestamped references  

---

# 🎯 Key Technologies
- **FastAPI** — backend server
- **FAISS** — vector similarity search
- **yt-dlp** — robust transcript extraction
- **OpenAI models** — embeddings + generation
- **Chrome Extensions API**
- **Shadow DOM** — isolation for UI & CSS

---

# 📝 Notes
- The extension works entirely locally using the local backend
- Backend must be running for extension to function

---

# ✔ Architecture is final, stable & production-ready.
