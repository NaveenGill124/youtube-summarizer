# Project Architecture

## Folder Structure
```
youtube-summarizer/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables (API keys)
│   └── vectorstores/          # FAISS indexes (gitignored)
│       └── {video_id}/
│           ├── index.faiss     # Vector embeddings
│           ├── chunks.json     # Text chunks
│           ├── metadata.json   # Timestamp metadata
│           └── transcript.json # Original transcript
│
├── Youtube_extension/
│   ├── manifest.json          # Extension configuration
│   ├── content_script.js      # Injected overlay UI
│   ├── popup.html             # Extension popup (unused)
│   ├── popup.js               # Popup logic (unused)
│   └── icons/                 # Extension icons
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
│
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
└── ARCHITECTURE.md           # This file
```

## Data Flow

1. **Video Detection**: Content script detects YouTube video ID
2. **Transcript Extraction**: Backend uses yt-dlp or HTML parsing
3. **Chunking**: Text split into 2-minute chunks with overlap
4. **Embedding**: OpenAI generates vector embeddings
5. **Storage**: FAISS stores vectors for fast similarity search
6. **Query**: User asks question → vectors searched → context retrieved
7. **Generation**: GPT-4o-mini generates answer from context
8. **Display**: Formatted answer shown in overlay

## Key Technologies

- **FAISS**: Facebook's similarity search library
- **RAG**: Retrieval Augmented Generation pattern
- **Shadow DOM**: CSS isolation for extension UI
- **yt-dlp**: Robust YouTube transcript downloader