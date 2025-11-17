# 🎥 YouTube Summarizer - AI-Powered Video Analysis Chrome Extension

An intelligent Chrome extension that uses RAG (Retrieval Augmented Generation) to provide instant summaries, key points, and Q&A for YouTube videos.

## ✨ Features

- 🤖 AI-powered video summarization
- 🔍 Semantic search through video transcripts
- ⏱️ Clickable timestamps to jump to specific parts
- 💬 Natural language Q&A about video content
- 🎯 Key points and learnings extraction
- 📊 Smart chunking with time-based context

## 🏗️ Architecture
```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Chrome Extension│ ───► │  FastAPI Backend │ ───► │   OpenAI    │
│  (Frontend)     │◄──── │   (RAG Pipeline) │◄──── │   API       │
└─────────────────┘      └──────────────────┘      └─────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  FAISS Vector   │
                         │     Store       │
                         └─────────────────┘
```

## 🚀 Tech Stack

**Frontend:**
- Chrome Extension APIs
- Shadow DOM for isolation
- Vanilla JavaScript

**Backend:**
- FastAPI (Python)
- OpenAI GPT-4o-mini
- OpenAI text-embedding-3-small
- FAISS (vector similarity search)
- yt-dlp (transcript extraction)

## 📦 Installation

### Prerequisites
- Python 3.8+
- Chrome Browser
- OpenAI API Key

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/youtube-summarizer.git
cd youtube-summarizer
```

2. Create virtual environment:
```bash
python -m venv mini
source mini/bin/activate  # On Windows: mini\Scripts\activate
```

3. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Create `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. Run the backend:
```bash
python main.py
```

The server will start at `http://localhost:8000`

### Chrome Extension Setup

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `Youtube_extension` folder
5. The extension is now installed!

## 🎯 Usage

1. Open any YouTube video
2. Click the floating "Y" button (bottom right)
3. Click "Summary" for a quick overview
4. Click "Key Points" for detailed learnings
5. Or ask any question about the video!

## 📊 API Endpoints

- `POST /ingest` - Index a video transcript
- `POST /query` - Query indexed video
- `GET /stats/{video_id}` - Get video statistics
- `DELETE /delete/{video_id}` - Remove indexed video

## 🔧 Configuration

Edit `backend/main.py` to adjust:
- `chunk_duration` - Time-based chunk size (default: 120s)
- `overlap_duration` - Overlap between chunks (default: 20s)
- `k` - Number of chunks to retrieve (default: 6-10)

## 📝 License

MIT License - feel free to use this project!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Known Limitations

- Requires videos to have English captions/transcripts
- Some networks may block transcript extraction
- OpenAI API costs apply based on usage

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini and embeddings API
- FAISS by Facebook Research
- yt-dlp community

---

Made with ❤️ by [Naveen Gill]