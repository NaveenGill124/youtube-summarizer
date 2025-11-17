# import requests
# from youtube_transcript_api import _api

# # Force YouTube to not block the request
# old_get = requests.get

# def patched_get(url, *args, **kwargs):
#     headers = kwargs.get("headers", {})
#     headers["User-Agent"] = (
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#         "AppleWebKit/537.36 (KHTML, like Gecko) "
#         "Chrome/120.0.0.0 Safari/537.36"
#     )
#     kwargs["headers"] = headers
#     return old_get(url, *args, **kwargs)

# requests.get = patched_get

# from pathlib import Path

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from youtube_transcript_api import YouTubeTranscriptApi
# from dotenv import load_dotenv
# import os, json
# import openai
# import numpy as np
# import faiss
# from pathlib import Path
# from fastapi.middleware.cors import CORSMiddleware



# import requests
# import json


# def get_transcript_mobile(video_id):
#     url = "https://www.youtube.com/youtubei/v1/get_transcript?key=AIzaSyAO_FJ2SlqU0GtYq33QAiFYa_xXc1GdC0c"
#     headers = {
#         "User-Agent": "com.google.android.youtube/17.31.35 (Linux; Android 11) gzip",
#         "Content-Type": "application/json"
#     }
#     body = {
#         "context": {
#             "client": {
#                 "clientName": "ANDROID",
#                 "clientVersion": "17.31.35"
#             }
#         },
#         "videoId": video_id
#     }

#     r = requests.post(url, headers=headers, json=body)
#     data = r.json()

#     segments = []
#     try:
#         actions = data["actions"][0]["updateEngagementPanelAction"]["content"]["transcriptRenderer"]["body"]["transcriptBodyRenderer"]["cueGroups"]
#         for entry in actions:
#             cue = entry["transcriptCueGroupRenderer"]["cues"][0]["transcriptCueRenderer"]
#             segments.append({"text": cue.get("cue", "")})
#     except Exception as e:
#         raise Exception("Transcript not available")

#     return segments





# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# app = FastAPI()
# BASE_DIR = Path(__file__).parent
# VECTOR_DIR = BASE_DIR / "vectorstores"
# VECTOR_DIR.mkdir(exist_ok=True)

# EMBED_DIM = 1536  # text-embedding-3-small

# # For development: allow all origins (be strict in production!)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],            # change to specific origins in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



# # ============================
# # Request Models
# # ============================
# class IngestRequest(BaseModel):
#     video_id: str

# class QueryRequest(BaseModel):
#     video_id: str
#     question: str

# # ============================
# # Helper functions
# # ============================

# def chunk_text(text, chunk_size=200, overlap=50):
#     words = text.split()
#     chunks = []
#     i = 0
#     while i < len(words):
#         chunk = words[i:i+chunk_size]
#         chunks.append(" ".join(chunk))
#         i += chunk_size - overlap
#     return chunks

# def embed_texts(texts):
#     res = openai.Embedding.create(model="text-embedding-3-small", input=texts)
#     return np.array([d["embedding"] for d in res["data"]], dtype="float32")

# # ============================
# # API Endpoints
# # ============================

# @app.get("/")
# async def root():
#     return {"status": "ok", "msg": "Backend running"}

# import requests
# import re
# import json

# def fetch_transcript_pbj(video_id):
#     url = f"https://www.youtube.com/watch?v={video_id}&pbj=1"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
#         "X-YouTube-Client-Name": "1",
#         "X-YouTube-Client-Version": "2.20201021.03.00",
#         "Accept-Language": "en-US,en;q=0.9",
#     }

#     r = requests.get(url, headers=headers)
#     if r.status_code != 200:
#         raise Exception(f"Failed to fetch page: {r.status_code}")

#     text = r.text

#     # Extract transcript JSON
#     match = re.search(r'"transcriptRenderer":({.*?})', text)
#     if not match:
#         raise Exception("No transcriptRenderer found (YouTube transcript disabled or protected).")

#     data = json.loads(match.group(1))

#     # Extract segments
#     segments = []
#     for entry in data["body"]["transcriptBodyRenderer"]["cueGroups"]:
#         seg = entry["transcriptCueGroupRenderer"]["cues"][0]["transcriptCueRenderer"]
#         segments.append({
#             "text": seg.get("cue", ""),
#         })

#     return segments



# @app.post("/ingest")
# async def ingest_video(req: IngestRequest):
#     vid = req.video_id
#     vdir = VECTOR_DIR / vid
#     if (vdir / "index.faiss").exists():
#         return {"status": "already_indexed", "video_id": vid}

#     try:
#         transcript = get_transcript_mobile(vid)
#         print("DEBUG: MOBILE API WORKING")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Transcript not found: {e}")


#     full_text = " ".join([seg["text"] for seg in transcript])
#     chunks = chunk_text(full_text)

#     embeddings = embed_texts(chunks)
#     faiss.normalize_L2(embeddings)

#     index = faiss.IndexFlatIP(EMBED_DIM)
#     index.add(embeddings)

#     # Save data
#     vdir.mkdir(exist_ok=True)
#     faiss.write_index(index, str(vdir / "index.faiss"))
#     with open(vdir / "chunks.json", "w") as f:
#         json.dump(chunks, f)

#     return {"status": "ok", "chunks": len(chunks), "video_id": vid}


# @app.post("/query")
# async def query_video(req: QueryRequest):
#     vid = req.video_id
#     q = req.question
#     vdir = VECTOR_DIR / vid

#     if not (vdir / "index.faiss").exists():
#         raise HTTPException(status_code=400, detail="Video not ingested yet. Please click 'Re-index' first.")

#     # Load FAISS and chunks
#     index = faiss.read_index(str(vdir / "index.faiss"))
#     with open(vdir / "chunks.json") as f:
#         chunks = json.load(f)

#     q_emb = embed_texts([q])
#     faiss.normalize_L2(q_emb)
#     D, I = index.search(q_emb, k=min(5, len(chunks)))

#     retrieved_chunks = [chunks[i] for i in I[0]]
#     context = "\n\n".join(retrieved_chunks)

#     system_prompt = "You are an AI assistant that answers questions using the given transcript excerpts. Be concise and factual."
#     user_prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             max_tokens=300,
#             temperature=0.2,
#         )
#         answer = response["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"LLM error: {e}")

#     return {"answer": answer, "retrieved_count": len(retrieved_chunks)}
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import tempfile

import requests
import numpy as np
import faiss
import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()
BASE_DIR = Path(__file__).parent
VECTOR_DIR = BASE_DIR / "vectorstores"
VECTOR_DIR.mkdir(exist_ok=True)

EMBED_DIM = 1536  # text-embedding-3-small

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Request Models
# ============================
class IngestRequest(BaseModel):
    video_id: str

class QueryRequest(BaseModel):
    video_id: str
    question: str

# ============================
# HTML-Based Transcript Extractor
# ============================

def extract_json_from_html(html):
    """Extract ytInitialPlayerResponse JSON from YouTube HTML"""
    pattern = r"var ytInitialPlayerResponse\s*=\s*({.+?});\s*(?:var|</script>)"
    match = re.search(pattern, html, re.DOTALL)
    
    if not match:
        # Try alternative pattern
        pattern = r"ytInitialPlayerResponse\s*=\s*({.+?});"
        match = re.search(pattern, html, re.DOTALL)
    
    if not match:
        raise Exception("ytInitialPlayerResponse not found in HTML")
    
    json_str = match.group(1)
    return json.loads(json_str)


def get_transcript_ytdlp(video_id):
    """
    Use yt-dlp to extract transcript - most reliable method
    """
    try:
        # Use yt-dlp to get subtitles
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "--output", "%(id)s",
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"yt-dlp failed: {result.stderr}")
            
            # Find the subtitle file
            subtitle_file = Path(tmpdir) / f"{video_id}.en.json3"
            
            if not subtitle_file.exists():
                # Try without language code
                subtitle_files = list(Path(tmpdir).glob(f"{video_id}*.json3"))
                if subtitle_files:
                    subtitle_file = subtitle_files[0]
                else:
                    raise Exception("No subtitle file created by yt-dlp")
            
            # Parse the subtitle file
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = []
            if "events" in data:
                for event in data["events"]:
                    if "segs" in event:
                        text_parts = []
                        for seg in event["segs"]:
                            if "utf8" in seg:
                                text_parts.append(seg["utf8"])
                        
                        if text_parts:
                            text = " ".join(text_parts).strip()
                            if text:
                                segments.append({"text": text})
            
            if not segments:
                raise Exception("No segments found in subtitle file")
            
            return segments
            
    except FileNotFoundError:
        raise Exception("yt-dlp is not installed. Install with: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        raise Exception("yt-dlp timed out")
    except Exception as e:
        raise Exception(f"yt-dlp extraction failed: {e}")


def get_transcript_from_html(video_id):
    """
    Extract transcript by parsing YouTube HTML and downloading captions.
    This method bypasses API restrictions and works in restricted networks.
    """
    # 1. Fetch YouTube watch page HTML
    url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch YouTube page: {response.status_code}")
    
    html = response.text
    
    # 2. Extract ytInitialPlayerResponse JSON
    try:
        player_data = extract_json_from_html(html)
    except Exception as e:
        raise Exception(f"Failed to parse player data: {e}")
    
    # 3. Get caption tracks
    try:
        captions = player_data["captions"]["playerCaptionsTracklistRenderer"]["captionTracks"]
    except KeyError:
        raise Exception("No captions available for this video (transcripts disabled by uploader)")
    
    if not captions:
        raise Exception("Caption tracks list is empty")
    
    # 4. Find English transcript or use first available
    transcript_url = None
    for track in captions:
        lang_code = track.get("languageCode", "")
        if "en" in lang_code.lower():
            transcript_url = track["baseUrl"]
            break
    
    # If no English found, fail explicitly
    if transcript_url is None:
        available_langs = [t.get('languageCode', 'unknown') for t in captions]
        raise Exception(f"No English ('en') caption track found. Available languages: {available_langs}")
    
    # 5. Add format parameter to get JSON instead of XML (more reliable)
    if "fmt=json3" not in transcript_url:
        transcript_url += "&fmt=json3"
    
    # 6. Download transcript
    try:
        transcript_response = requests.get(transcript_url, headers=headers, timeout=10)
        if transcript_response.status_code != 200:
            raise Exception(f"Failed to download transcript: {transcript_response.status_code}")
        
        # Check if response is empty
        if not transcript_response.text or len(transcript_response.text) < 10:
            raise Exception("Transcript response is empty - likely regional blocking")
        
        transcript_data = transcript_response.json()
    except json.JSONDecodeError:
        # Fallback to XML parsing
        return parse_xml_transcript(transcript_response.text)
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")
    
    # 7. Extract text from JSON format
    segments = []
    
    if "events" in transcript_data:
        for event in transcript_data["events"]:
            if "segs" in event:
                text_parts = []
                for seg in event["segs"]:
                    if "utf8" in seg:
                        text_parts.append(seg["utf8"])
                
                if text_parts:
                    text = " ".join(text_parts).strip()
                    if text:
                        segments.append({"text": text})
    
    if not segments:
        raise Exception("No text segments found in transcript")
    
    return segments


def parse_xml_transcript(xml_text):
    """Fallback XML parser for transcript"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise Exception(f"Failed to parse transcript XML: {e}")
    
    segments = []
    for element in root:
        if element.tag == "text":
            text = element.text or ""
            # Clean up text
            text = text.replace("\n", " ").strip()
            # Decode HTML entities
            text = text.replace("&amp;", "&")
            text = text.replace("&lt;", "<")
            text = text.replace("&gt;", ">")
            text = text.replace("&quot;", '"')
            text = text.replace("&#39;", "'")
            
            if text:
                segments.append({"text": text})
    
    return segments


# ============================
# Helper Functions
# ============================

def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


from openai import OpenAI
client = OpenAI()

def embed_texts(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in res.data], dtype="float32")


# ============================
# API Endpoints
# ============================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "YouTube Transcript RAG Backend Running",
        "version": "2.0-html-extractor"
    }


@app.post("/ingest")
async def ingest_video(req: IngestRequest):
    """
    Ingest a YouTube video by extracting transcript,
    chunking, embedding, and storing in FAISS
    """
    vid = req.video_id
    vdir = VECTOR_DIR / vid
    
    # Check if already indexed
    if (vdir / "index.faiss").exists():
        return {
            "status": "already_indexed",
            "video_id": vid,
            "message": "This video is already in the database"
        }
    
    # Extract transcript using multiple fallback methods
    transcript = None
    errors = []
    
    # Method 1: Try yt-dlp first (most reliable)
    try:
        transcript = get_transcript_ytdlp(vid)
        print(f"✅ Successfully extracted transcript using yt-dlp")
    except Exception as e:
        errors.append(f"yt-dlp: {str(e)}")
        print(f"⚠️ yt-dlp failed: {e}")
    
    # Method 2: Try HTML extraction
    if not transcript:
        try:
            transcript = get_transcript_from_html(vid)
            print(f"✅ Successfully extracted transcript using HTML method")
        except Exception as e:
            errors.append(f"HTML: {str(e)}")
            print(f"⚠️ HTML method failed: {e}")
    
    if not transcript:
        error_msg = "Failed to extract transcript using all methods:\n" + "\n".join(errors)
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Combine all segments into full text
    full_text = " ".join([seg["text"] for seg in transcript])
    
    if not full_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Transcript is empty"
        )
    
    # Chunk the text
    chunks = chunk_text(full_text)
    
    # Generate embeddings
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    
    # Save to disk
    vdir.mkdir(exist_ok=True)
    faiss.write_index(index, str(vdir / "index.faiss"))
    
    with open(vdir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    return {
        "status": "success",
        "video_id": vid,
        "chunks": len(chunks),
        "message": f"Successfully indexed {len(chunks)} chunks"
    }


@app.post("/query")
async def query_video(req: QueryRequest):
    """
    Query a video using RAG:
    1. Retrieve relevant chunks from FAISS
    2. Use GPT-4o-mini to generate answer
    """
    vid = req.video_id
    question = req.question
    vdir = VECTOR_DIR / vid
    
    # Check if video is indexed
    if not (vdir / "index.faiss").exists():
        raise HTTPException(
            status_code=400,
            detail="Video not indexed yet. Please run /ingest first."
        )
    
    # Load FAISS index and chunks
    index = faiss.read_index(str(vdir / "index.faiss"))
    
    with open(vdir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Embed the question
    try:
        q_embedding = embed_texts([question])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed question: {str(e)}"
        )
    
    # Normalize for cosine similarity
    faiss.normalize_L2(q_embedding)
    
    # Search for top-k relevant chunks
    k = min(5, len(chunks))
    distances, indices = index.search(q_embedding, k)
    
    # Retrieve chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)
    
    # Generate answer using GPT-4o-mini
    system_prompt = (
        "You are an AI assistant that answers questions based on YouTube video transcripts. "
        "Use only the provided context to answer. Be concise, accurate, and factual. "
        "If the context doesn't contain enough information, say so."
    )
    
    user_prompt = f"Context from video transcript:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)
        answer = response.choices[0].message.content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )
    
    return {
        "answer": answer,
        "retrieved_chunks": len(retrieved_chunks),
        "relevance_scores": distances[0].tolist()
    }


@app.delete("/delete/{video_id}")
async def delete_video(video_id: str):
    """Delete a video's index from the database"""
    vdir = VECTOR_DIR / video_id
    
    if not vdir.exists():
        raise HTTPException(
            status_code=404,
            detail="Video not found in database"
        )
    
    # Delete all files
    import shutil
    shutil.rmtree(vdir)
    
    return {
        "status": "success",
        "message": f"Deleted index for video {video_id}"
    }


@app.get("/debug/{video_id}")
async def debug_transcript(video_id: str):
    """Debug endpoint to see what's happening with transcript extraction"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    try:
        # Fetch HTML
        response = requests.get(url, headers=headers, timeout=10)
        html = response.text
        
        # Extract player data
        player_data = extract_json_from_html(html)
        
        # Get captions info
        captions_info = player_data.get("captions", {})
        
        if "playerCaptionsTracklistRenderer" not in captions_info:
            return {
                "status": "no_captions",
                "message": "No captions found in player data",
                "available_keys": list(captions_info.keys())
            }
        
        caption_tracks = captions_info["playerCaptionsTracklistRenderer"].get("captionTracks", [])
        
        # Get first caption URL
        if caption_tracks:
            first_track = caption_tracks[0]
            transcript_url = first_track["baseUrl"]
            
            # Try to download
            xml_response = requests.get(transcript_url, headers=headers, timeout=10)
            
            return {
                "status": "success",
                "video_id": video_id,
                "caption_tracks_count": len(caption_tracks),
                "first_track_language": first_track.get("languageCode"),
                "transcript_url": transcript_url,
                "xml_status_code": xml_response.status_code,
                "xml_length": len(xml_response.text),
                "xml_preview": xml_response.text[:500]
            }
        else:
            return {
                "status": "no_tracks",
                "message": "Caption tracks list is empty"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 YouTube Transcript RAG Backend Starting...")
    print("="*50)
    print(f"📍 Access at: http://localhost:8000")
    print(f"📍 Docs at: http://localhost:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    
#main.py file content which worked there

# import re
# import json
# import xml.etree.ElementTree as ET
# from pathlib import Path
# import subprocess
# import tempfile

# import requests
# import numpy as np
# import faiss
# import openai
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()
# from openai import OpenAI
# # This is the ONLY client you need.
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# app = FastAPI()
# BASE_DIR = Path(__file__).parent
# VECTOR_DIR = BASE_DIR / "vectorstores"
# VECTOR_DIR.mkdir(exist_ok=True)

# EMBED_DIM = 1536  # text-embedding-3-small

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================
# # Request Models
# # ============================
# class IngestRequest(BaseModel):
#     video_id: str

# class QueryRequest(BaseModel):
#     video_id: str
#     question: str

# # ============================
# # HTML-Based Transcript Extractor
# # ============================

# def extract_json_from_html(html):
#     """Extract ytInitialPlayerResponse JSON from YouTube HTML"""
#     pattern = r"var ytInitialPlayerResponse\s*=\s*({.+?});\s*(?:var|</script>)"
#     match = re.search(pattern, html, re.DOTALL)
    
#     if not match:
#         # Try alternative pattern
#         pattern = r"ytInitialPlayerResponse\s*=\s*({.+?});"
#         match = re.search(pattern, html, re.DOTALL)
    
#     if not match:
#         raise Exception("ytInitialPlayerResponse not found in HTML")
    
#     json_str = match.group(1)
#     return json.loads(json_str)


# def get_transcript_ytdlp(video_id):
#     """
#     Use yt-dlp to extract transcript - most reliable method
#     """
#     try:
#         cmd = [
#             "yt-dlp",
#             "--skip-download",
#             "--write-auto-subs",
#             "--sub-lang", "en",
#             "--sub-format", "json3",
#             "--output", "%(id)s",
#             # REMOVED: The "--cookies-from-browser" line that caused the crash
#             f"https://www.youtube.com/watch?v={video_id}"
#         ]

#         with tempfile.TemporaryDirectory() as tmpdir:
#             result = subprocess.run(
#                 cmd,
#                 cwd=tmpdir,
#                 capture_output=True,
#                 text=True,
#                 timeout=60
#             )

#             if result.returncode != 0:
#                 raise Exception(f"yt-dlp failed: {result.stderr}")

#             subtitle_file = Path(tmpdir) / f"{video_id}.en.json3"

#             if not subtitle_file.exists():
#                 subtitle_files = list(Path(tmpdir).glob(f"{video_id}*.json3"))
#                 if subtitle_files:
#                     subtitle_file = subtitle_files[0]
#                 else:
#                     raise Exception("No subtitle file created by yt-dlp")

#             with open(subtitle_file, "r", encoding="utf-8") as f:
#                 data = json.load(f)

#             segments = []
#             if "events" in data:
#                 for event in data["events"]:
#                     if "segs" in event:
#                         text_parts = [
#                             seg["utf8"]
#                             for seg in event["segs"]
#                             if "utf8" in seg
#                         ]
#                         if text_parts:
#                             text = " ".join(text_parts).strip()
#                             if text:
#                                 segments.append({"text": text})

#             if not segments:
#                 raise Exception("No segments found in subtitle file")

#             return segments

#     except FileNotFoundError:
#         raise Exception("yt-dlp is not installed. Install with: pip install yt-dlp")
#     except subprocess.TimeoutExpired:
#         raise Exception("yt-dlp timed out")
#     except Exception as e:
#         raise Exception(f"yt-dlp extraction failed: {e}")



# def get_transcript_from_html(video_id):
#     """
#     Extract transcript by parsing YouTube HTML and downloading captions.
#     This method bypasses API restrictions and works in restricted networks.
#     """
#     # 1. Fetch YouTube watch page HTML
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#         "Accept-Language": "en-US,en;q=0.9"
#     }
    
#     response = requests.get(url, headers=headers, timeout=10)
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch YouTube page: {response.status_code}")
    
#     html = response.text
    
#     # 2. Extract ytInitialPlayerResponse JSON
#     try:
#         player_data = extract_json_from_html(html)
#     except Exception as e:
#         raise Exception(f"Failed to parse player data: {e}")
    
#     # 3. Get caption tracks
#     try:
#         captions = player_data["captions"]["playerCaptionsTracklistRenderer"]["captionTracks"]
#     except KeyError:
#         raise Exception("No captions available for this video (transcripts disabled by uploader)")
    
#     if not captions:
#         raise Exception("Caption tracks list is empty")
    
#     # 4. Find English transcript or use first available
#     transcript_url = None
#     for track in captions:
#         lang_code = track.get("languageCode", "")
#         if "en" in lang_code.lower():
#             transcript_url = track["baseUrl"]
#             break
    
#     # <--- FIX 1: This logic is now corrected --->
#     # If no English found, fail explicitly
#     if transcript_url is None:
#         available_langs = [t.get('languageCode', 'unknown') for t in captions]
#         raise Exception(f"No English ('en') caption track found. Available languages: {available_langs}")
#     # <--- END FIX 1 --->
    
#     # 5. Add format parameter to get JSON instead of XML (more reliable)
#     if "fmt=json3" not in transcript_url:
#         transcript_url += "&fmt=json3"
    
#     # 6. Download transcript
#     try:
#         transcript_response = requests.get(transcript_url, headers=headers, timeout=10)
#         if transcript_response.status_code != 200:
#             raise Exception(f"Failed to download transcript: {transcript_response.status_code}")
        
#         # Check if response is empty
#         if not transcript_response.text or len(transcript_response.text) < 10:
#             raise Exception("Transcript response is empty - likely regional blocking")
        
#         transcript_data = transcript_response.json()
#     except json.JSONDecodeError:
#         # Fallback to XML parsing
#         return parse_xml_transcript(transcript_response.text)
#     except Exception as e:
#         raise Exception(f"Failed to fetch transcript: {e}")
    
#     # 7. Extract text from JSON format
#     segments = []
    
#     if "events" in transcript_data:
#         for event in transcript_data["events"]:
#             if "segs" in event:
#                 text_parts = []
#                 for seg in event["segs"]:
#                     if "utf8" in seg:
#                         text_parts.append(seg["utf8"])
                
#                 if text_parts:
#                     text = " ".join(text_parts).strip()
#                     if text:
#                         segments.append({"text": text})
    
#     if not segments:
#         raise Exception("No text segments found in transcript")
    
#     return segments


# def parse_xml_transcript(xml_text):
#     """Fallback XML parser for transcript"""
#     try:
#         root = ET.fromstring(xml_text)
#     except ET.ParseError as e:
#         raise Exception(f"Failed to parse transcript XML: {e}")
    
#     segments = []
#     for element in root:
#         if element.tag == "text":
#             text = element.text or ""
#             # Clean up text
#             text = text.replace("\n", " ").strip()
#             # Decode HTML entities
#             text = text.replace("&amp;", "&")
#             text = text.replace("&lt;", "<")
#             text = text.replace("&gt;", ">")
#             text = text.replace("&quot;", '"')
#             text = text.replace("&#39;", "'")
            
#             if text:
#                 segments.append({"text": text})
    
#     return segments


# # ============================
# # Helper Functions
# # ============================

# def chunk_text(text, chunk_size=300, overlap=100):
#     """Better chunking with more overlap for context"""
#     words = text.split()
#     chunks = []
#     i = 0
#     while i < len(words):
#         chunk = words[i:i+chunk_size]
#         chunks.append(" ".join(chunk))
#         i += chunk_size - overlap
#     return chunks


# # <--- FIX 2: Deleted the redundant 'from openai import OpenAI' and 'client = OpenAI()' lines --->

# def embed_texts(texts):
#     # This function now uses the 'client' defined at the top of the file
#     res = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=texts
#     )
#     return np.array([d.embedding for d in res.data], dtype="float32")


# # ============================
# # API Endpoints
# # ============================

# @app.get("/")
# async def root():
#     return {
#         "status": "ok",
#         "message": "YouTube Transcript RAG Backend Running",
#         "version": "2.0-html-extractor"
#     }


# @app.post("/ingest")
# async def ingest_video(req: IngestRequest):
#     """
#     Ingest a YouTube video by extracting transcript,
#     chunking, embedding, and storing in FAISS
#     """
#     vid = req.video_id
#     vdir = VECTOR_DIR / vid
    
#     # Check if already indexed
#     if (vdir / "index.faiss").exists():
#         return {
#             "status": "already_indexed",
#             "video_id": vid,
#             "message": "This video is already in the database"
#         }
    
#     # Extract transcript using multiple fallback methods
#     transcript = None
#     errors = []
    
#     # Method 1: Try yt-dlp first (most reliable)
#     try:
#         transcript = get_transcript_ytdlp(vid)
#         print(f"✅ Successfully extracted transcript using yt-dlp")
#     except Exception as e:
#         errors.append(f"yt-dlp: {str(e)}")
#         print(f"⚠️ yt-dlp failed: {e}")

#     # Method 2: Try HTML extraction
#     if not transcript:
#         try:
#             transcript = get_transcript_from_html(vid)
#             print(f"✅ Successfully extracted transcript using HTML method")
#         except Exception as e:
#             errors.append(f"HTML: {str(e)}")
#             print(f"⚠️ HTML method failed: {e}")

    
#     if not transcript:
#         error_msg = "Failed to extract transcript using all methods:\n" + "\n".join(errors)
#         raise HTTPException(status_code=400, detail=error_msg)
    
#     # Combine all segments into full text
#     full_text = " ".join([seg["text"] for seg in transcript])
    
#     if not full_text.strip():
#         raise HTTPException(
#             status_code=400,
#             detail="Transcript is empty"
#         )
    
#     # Chunk the text
#     chunks = chunk_text(full_text)
    
#     # Generate embeddings
#     try:
#         embeddings = embed_texts(chunks)
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to generate embeddings: {str(e)}"
#         )
    
#     # Normalize embeddings for cosine similarity
#     faiss.normalize_L2(embeddings)
    
#     # Create FAISS index
#     index = faiss.IndexFlatIP(EMBED_DIM)
#     index.add(embeddings)
    
#     # Save to disk
#     vdir.mkdir(exist_ok=True)
#     faiss.write_index(index, str(vdir / "index.faiss"))
    
#     with open(vdir / "chunks.json", "w", encoding="utf-8") as f:
#         json.dump(chunks, f, ensure_ascii=False, indent=2)
    
#     return {
#         "status": "success",
#         "video_id": vid,
#         "chunks": len(chunks),
#         "message": f"Successfully indexed {len(chunks)} chunks"
#     }


# @app.post("/query")
# async def query_video(req: QueryRequest):
#     """
#     Query a video using RAG:
#     1. Retrieve relevant chunks from FAISS
#     2. Use GPT-4o-mini to generate answer
#     """
#     vid = req.video_id
#     question = req.question
#     vdir = VECTOR_DIR / vid
    
#     # Check if video is indexed
#     if not (vdir / "index.faiss").exists():
#         raise HTTPException(
#             status_code=400,
#             detail="Video not indexed yet. Please run /ingest first."
#         )
    
#     # Load FAISS index and chunks
#     index = faiss.read_index(str(vdir / "index.faiss"))
    
#     with open(vdir / "chunks.json", "r", encoding="utf-8") as f:
#         chunks = json.load(f)
    
#     # Embed the question
#     try:
#         q_embedding = embed_texts([question])
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to embed question: {str(e)}"
#         )
    
#     # Normalize for cosine similarity
#     faiss.normalize_L2(q_embedding)
    
#     # Search for top-k relevant chunks
#     k = min(5, len(chunks))
#     distances, indices = index.search(q_embedding, k)
    
#     # Retrieve chunks
#     retrieved_chunks = [chunks[i] for i in indices[0]]
#     context = "\n\n".join(retrieved_chunks)
    
#     # Generate answer using GPT-4o-mini
#     system_prompt = (
#         "You are an AI assistant that answers questions based on YouTube video transcripts. "
#         "Use only the provided context to answer. Be concise, accurate, and factual. "
#         "If the context doesn't contain enough information, say so."
#     )
    
#     user_prompt = f"Context from video transcript:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ]
#         )
#         answer = response.choices[0].message.content

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to generate answer: {str(e)}"
#         )
    
#     return {
#         "answer": answer,
#         "retrieved_chunks": len(retrieved_chunks),
#         "relevance_scores": distances[0].tolist()
#     }


# @app.delete("/delete/{video_id}")
# async def delete_video(video_id: str):
#     """Delete a video's index from the database"""
#     vdir = VECTOR_DIR / video_id
    
#     if not vdir.exists():
#         raise HTTPException(
#             status_code=404,
#             detail="Video not found in database"
#         )
    
#     # Delete all files
#     import shutil
#     shutil.rmtree(vdir)
    
#     return {
#         "status": "success",
#         "message": f"Deleted index for video {video_id}"
#     }

# @app.get("/debug/{video_id}")
# async def debug_transcript(video_id: str):
#     """Debug endpoint to see transcript extraction + caption meta"""
#     info = {
#         "video_id": video_id,
#         "yt_dlp": None,
#         "html_method": None,
#     }

#     # Try yt-dlp
#     try:
#         segments = get_transcript_ytdlp(video_id)
#         info["yt_dlp"] = {
#             "status": "success",
#             "segments_count": len(segments),
#             "sample": segments[:3],
#         }
#     except Exception as e:
#         info["yt_dlp"] = {
#             "status": "error",
#             "error": str(e),
#         }

#     # Try HTML method
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#         "Accept-Language": "en-US,en;q=0.9",
#     }

#     try:
#         response = requests.get(url, headers=headers, timeout=10)
#         html = response.text
#         player_data = extract_json_from_html(html)
#         captions_info = player_data.get("captions", {})
#         tracklist = captions_info.get("playerCaptionsTracklistRenderer", {})
#         tracks = tracklist.get("captionTracks", [])

#         if not tracks:
#             info["html_method"] = {
#                 "status": "no_tracks",
#                 "caption_tracks_count": 0,
#                 "available_keys": list(captions_info.keys()),
#             }
#         else:
#             first_track = tracks[0]
#             transcript_url = first_track["baseUrl"]

#             # Try adding fmt=json3 for debugging too
#             if "fmt=json3" not in transcript_url:
#                 transcript_url_with_fmt = transcript_url + "&fmt=json3"
#             else:
#                 transcript_url_with_fmt = transcript_url

#             xml_resp = requests.get(transcript_url_with_fmt, headers=headers, timeout=10)

#             info["html_method"] = {
#                 "status": "success",
#                 "caption_tracks_count": len(tracks),
#                 "first_track_language": first_track.get("languageCode"),
#                 "transcript_url": transcript_url_with_fmt,
#                 "xml_status_code": xml_resp.status_code,
#                 "xml_length": len(xml_resp.text),
#                 "xml_preview": xml_resp.text[:500],
#                 "xml_headers": dict(xml_resp.headers),
#             }

#     except Exception as e:
#         info["html_method"] = {
#             "status": "error",
#             "error": str(e),
#         }

#     return info


# if __name__ == "__main__":
#     import uvicorn
#     print("\n" + "="*50)
#     print("🚀 YouTube Transcript RAG Backend Starting...")
#     print("="*50)
#     print(f"📍 Access at: http://localhost:8000")
#     print(f"📍 Docs at: http://localhost:8000/docs")
#     print("="*50 + "\n")
#     uvicorn.run(app, host="127.0.0.1", port=8000)

#-----------------------------------------------------------------------------$\
# 2nd trial attempt


# import re
# import json
# import xml.etree.ElementTree as ET
# from pathlib import Path
# import subprocess
# import tempfile
# from typing import List, Dict, Optional

# import requests
# import numpy as np
# import faiss
# from openai import OpenAI
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# # Initialize OpenAI client (NEW v1.0+ syntax)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app = FastAPI()
# BASE_DIR = Path(__file__).parent
# VECTOR_DIR = BASE_DIR / "vectorstores"
# VECTOR_DIR.mkdir(exist_ok=True)

# EMBED_DIM = 1536  # text-embedding-3-small

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================
# # Request Models
# # ============================
# class IngestRequest(BaseModel):
#     video_id: str

# class QueryRequest(BaseModel):
#     video_id: str
#     question: str

# # ============================
# # Enhanced Transcript Extraction
# # ============================

# def extract_json_from_html(html):
#     """Extract ytInitialPlayerResponse JSON from YouTube HTML"""
#     pattern = r"var ytInitialPlayerResponse\s*=\s*({.+?});\s*(?:var|</script>)"
#     match = re.search(pattern, html, re.DOTALL)
    
#     if not match:
#         pattern = r"ytInitialPlayerResponse\s*=\s*({.+?});"
#         match = re.search(pattern, html, re.DOTALL)
    
#     if not match:
#         raise Exception("ytInitialPlayerResponse not found in HTML")
    
#     json_str = match.group(1)
#     return json.loads(json_str)


# def get_transcript_ytdlp(video_id):
#     """Use yt-dlp to extract transcript with timestamps"""
#     try:
#         cmd = [
#             "yt-dlp",
#             "--skip-download",
#             "--write-auto-subs",
#             "--sub-lang", "en",
#             "--sub-format", "json3",
#             "--output", "%(id)s",
#             f"https://www.youtube.com/watch?v={video_id}"
#         ]
        
#         with tempfile.TemporaryDirectory() as tmpdir:
#             result = subprocess.run(
#                 cmd,
#                 cwd=tmpdir,
#                 capture_output=True,
#                 text=True,
#                 timeout=60
#             )
            
#             if result.returncode != 0:
#                 raise Exception(f"yt-dlp failed: {result.stderr}")
            
#             subtitle_file = Path(tmpdir) / f"{video_id}.en.json3"
            
#             if not subtitle_file.exists():
#                 subtitle_files = list(Path(tmpdir).glob(f"{video_id}*.json3"))
#                 if subtitle_files:
#                     subtitle_file = subtitle_files[0]
#                 else:
#                     raise Exception("No subtitle file created by yt-dlp")
            
#             with open(subtitle_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
            
#             segments = []
#             if "events" in data:
#                 for event in data["events"]:
#                     if "segs" in event:
#                         text_parts = []
#                         for seg in event["segs"]:
#                             if "utf8" in seg:
#                                 text_parts.append(seg["utf8"])
                        
#                         if text_parts:
#                             text = " ".join(text_parts).strip()
#                             timestamp = event.get("tStartMs", 0) / 1000  # Convert to seconds
#                             if text:
#                                 segments.append({
#                                     "text": text,
#                                     "timestamp": timestamp
#                                 })
            
#             if not segments:
#                 raise Exception("No segments found in subtitle file")
            
#             return segments
            
#     except FileNotFoundError:
#         raise Exception("yt-dlp is not installed. Install with: pip install yt-dlp")
#     except subprocess.TimeoutExpired:
#         raise Exception("yt-dlp timed out")
#     except Exception as e:
#         raise Exception(f"yt-dlp extraction failed: {e}")


# def get_transcript_from_html(video_id):
#     """Fallback HTML-based extraction"""
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#         "Accept-Language": "en-US,en;q=0.9"
#     }
    
#     response = requests.get(url, headers=headers, timeout=10)
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch YouTube page: {response.status_code}")
    
#     html = response.text
    
#     try:
#         player_data = extract_json_from_html(html)
#     except Exception as e:
#         raise Exception(f"Failed to parse player data: {e}")
    
#     try:
#         captions = player_data["captions"]["playerCaptionsTracklistRenderer"]["captionTracks"]
#     except KeyError:
#         raise Exception("No captions available for this video")
    
#     if not captions:
#         raise Exception("Caption tracks list is empty")
    
#     transcript_url = None
#     for track in captions:
#         lang_code = track.get("languageCode", "")
#         if "en" in lang_code.lower():
#             transcript_url = track["baseUrl"]
#             break
    
#     if transcript_url is None:
#         transcript_url = captions[0]["baseUrl"]
    
#     if "fmt=json3" not in transcript_url:
#         transcript_url += "&fmt=json3"
    
#     try:
#         transcript_response = requests.get(transcript_url, headers=headers, timeout=10)
#         if transcript_response.status_code != 200:
#             raise Exception(f"Failed to download transcript: {transcript_response.status_code}")
        
#         if not transcript_response.text or len(transcript_response.text) < 10:
#             raise Exception("Transcript response is empty")
        
#         transcript_data = transcript_response.json()
#     except Exception as e:
#         raise Exception(f"Failed to fetch transcript: {e}")
    
#     segments = []
#     if "events" in transcript_data:
#         for event in transcript_data["events"]:
#             if "segs" in event:
#                 text_parts = []
#                 for seg in event["segs"]:
#                     if "utf8" in seg:
#                         text_parts.append(seg["utf8"])
                
#                 if text_parts:
#                     text = " ".join(text_parts).strip()
#                     timestamp = event.get("tStartMs", 0) / 1000
#                     if text:
#                         segments.append({
#                             "text": text,
#                             "timestamp": timestamp
#                         })
    
#     if not segments:
#         raise Exception("No text segments found in transcript")
    
#     return segments


# # ============================
# # Enhanced Chunking Strategy
# # ============================

# def format_timestamp(seconds):
#     """Convert seconds to MM:SS format"""
#     minutes = int(seconds // 60)
#     secs = int(seconds % 60)
#     return f"{minutes}:{secs:02d}"


# def create_smart_chunks(segments: List[Dict], chunk_duration=120, overlap_duration=20):
#     """
#     Create time-based chunks with overlap.
#     Better for maintaining context and allowing timestamp references.
#     """
#     if not segments:
#         return [], []
    
#     chunks = []
#     chunk_metadata = []
    
#     current_chunk = []
#     chunk_start = segments[0].get("timestamp", 0)
#     chunk_end = chunk_start + chunk_duration
    
#     for segment in segments:
#         timestamp = segment.get("timestamp", 0)
#         text = segment["text"]
        
#         # If we're past the chunk end, save current chunk and start new one
#         if timestamp > chunk_end and current_chunk:
#             chunk_text = " ".join(current_chunk)
#             chunks.append(chunk_text)
#             chunk_metadata.append({
#                 "start": chunk_start,
#                 "end": timestamp,
#                 "start_formatted": format_timestamp(chunk_start),
#                 "end_formatted": format_timestamp(timestamp)
#             })
            
#             # Start new chunk with overlap
#             overlap_start = chunk_end - overlap_duration
#             current_chunk = []
            
#             # Add overlapping segments
#             for prev_seg in segments:
#                 prev_time = prev_seg.get("timestamp", 0)
#                 if overlap_start <= prev_time < chunk_end:
#                     current_chunk.append(prev_seg["text"])
            
#             chunk_start = timestamp
#             chunk_end = chunk_start + chunk_duration
        
#         current_chunk.append(text)
    
#     # Add the last chunk
#     if current_chunk:
#         chunk_text = " ".join(current_chunk)
#         chunks.append(chunk_text)
#         chunk_metadata.append({
#             "start": chunk_start,
#             "end": segments[-1].get("timestamp", 0),
#             "start_formatted": format_timestamp(chunk_start),
#             "end_formatted": format_timestamp(segments[-1].get("timestamp", 0))
#         })
    
#     return chunks, chunk_metadata


# # ============================
# # Embeddings (FIXED - NEW OPENAI SYNTAX)
# # ============================

# def embed_texts(texts):
#     """Generate embeddings using OpenAI (v1.0+ syntax)"""
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=texts
#     )
#     return np.array([item.embedding for item in response.data], dtype="float32")


# # ============================
# # Enhanced Query with Better Prompts
# # ============================

# def get_enhanced_system_prompt():
#     """System prompt for better, more structured answers"""
#     return """You are an expert YouTube video analyst. Your job is to provide clear, accurate, and well-structured answers based on video transcripts.

# Guidelines:
# 1. Always cite specific information from the transcript
# 2. If names or specific details are mentioned, include them
# 3. Structure your answers with clear sections when appropriate
# 4. Include timestamps when relevant
# 5. If the transcript doesn't contain enough information, say so clearly
# 6. Be concise but comprehensive
# 7. For summary requests, provide:
#    - Main topic/theme
#    - Key speakers (if identifiable)
#    - Major points discussed
#    - Important takeaways or conclusions
# """


# def format_context_with_timestamps(chunks: List[str], metadata: List[Dict], indices: List[int]):
#     """Format retrieved chunks with timestamp information"""
#     formatted_parts = []
    
#     for idx in indices:
#         chunk = chunks[idx]
#         meta = metadata[idx]
#         formatted_parts.append(
#             f"[{meta['start_formatted']} - {meta['end_formatted']}]\n{chunk}"
#         )
    
#     return "\n\n---\n\n".join(formatted_parts)


# # ============================
# # API Endpoints
# # ============================

# @app.get("/")
# async def root():
#     return {
#         "status": "ok",
#         "message": "YouTube Transcript RAG Backend - Production Version",
#         "version": "3.0"
#     }


# @app.post("/ingest")
# async def ingest_video(req: IngestRequest):
#     """Ingest video with enhanced chunking and metadata"""
#     vid = req.video_id
#     vdir = VECTOR_DIR / vid
    
#     if (vdir / "index.faiss").exists():
#         # Load existing metadata to return info
#         with open(vdir / "metadata.json", "r", encoding="utf-8") as f:
#             existing_meta = json.load(f)
#         return {
#             "status": "already_indexed",
#             "video_id": vid,
#             "chunks": len(existing_meta),
#             "message": "Video already indexed"
#         }
    
#     # Extract transcript
#     transcript = None
#     errors = []
    
#     try:
#         transcript = get_transcript_ytdlp(vid)
#         print(f"✅ Successfully extracted transcript using yt-dlp")
#     except Exception as e:
#         errors.append(f"yt-dlp: {str(e)}")
#         print(f"⚠️ yt-dlp failed: {e}")
    
#     if not transcript:
#         try:
#             transcript = get_transcript_from_html(vid)
#             print(f"✅ Successfully extracted transcript using HTML method")
#         except Exception as e:
#             errors.append(f"HTML: {str(e)}")
#             print(f"⚠️ HTML method failed: {e}")
    
#     if not transcript:
#         error_msg = "Failed to extract transcript:\n" + "\n".join(errors)
#         raise HTTPException(status_code=400, detail=error_msg)
    
#     # Create smart chunks with timestamps
#     chunks, chunk_metadata = create_smart_chunks(transcript)
    
#     if not chunks:
#         raise HTTPException(status_code=400, detail="No chunks created from transcript")
    
#     # Generate embeddings
#     try:
#         embeddings = embed_texts(chunks)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")
    
#     faiss.normalize_L2(embeddings)
    
#     # Create FAISS index
#     index = faiss.IndexFlatIP(EMBED_DIM)
#     index.add(embeddings)
    
#     # Save everything
#     vdir.mkdir(exist_ok=True)
#     faiss.write_index(index, str(vdir / "index.faiss"))
    
#     with open(vdir / "chunks.json", "w", encoding="utf-8") as f:
#         json.dump(chunks, f, ensure_ascii=False, indent=2)
    
#     with open(vdir / "metadata.json", "w", encoding="utf-8") as f:
#         json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
    
#     # Store full transcript for reference
#     with open(vdir / "transcript.json", "w", encoding="utf-8") as f:
#         json.dump(transcript, f, ensure_ascii=False, indent=2)
    
#     return {
#         "status": "success",
#         "video_id": vid,
#         "chunks": len(chunks),
#         "duration": f"{chunk_metadata[-1]['end']:.0f}s",
#         "message": f"Successfully indexed {len(chunks)} chunks"
#     }


# @app.post("/query")
# async def query_video(req: QueryRequest):
#     """Query with enhanced retrieval and better prompting"""
#     vid = req.video_id
#     question = req.question
#     vdir = VECTOR_DIR / vid
    
#     if not (vdir / "index.faiss").exists():
#         raise HTTPException(
#             status_code=400,
#             detail="Video not indexed. Please run /ingest first."
#         )
    
#     # Load index, chunks, and metadata
#     index = faiss.read_index(str(vdir / "index.faiss"))
    
#     with open(vdir / "chunks.json", "r", encoding="utf-8") as f:
#         chunks = json.load(f)
    
#     with open(vdir / "metadata.json", "r", encoding="utf-8") as f:
#         metadata = json.load(f)
    
#     # Embed question
#     try:
#         q_embedding = embed_texts([question])
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to embed question: {str(e)}")
    
#     faiss.normalize_L2(q_embedding)
    
#     # Retrieve more chunks for better context (especially for summaries)
#     is_summary_request = any(word in question.lower() for word in ['summary', 'summarize', 'overview', 'about'])
#     k = min(10 if is_summary_request else 6, len(chunks))
    
#     distances, indices = index.search(q_embedding, k)
    
#     # Format context with timestamps
#     context = format_context_with_timestamps(chunks, metadata, indices[0])
    
#     # Enhanced prompt
#     system_prompt = get_enhanced_system_prompt()
    
#     user_prompt = f"""Based on the following excerpts from a YouTube video transcript (with timestamps), please answer the question.

# Transcript excerpts:
# {context}

# Question: {question}

# Instructions:
# - Provide a clear, well-structured answer
# - Reference specific parts of the transcript when relevant
# - Include timestamps if they help illustrate your points
# - If asking about people, identify them by name if mentioned in the transcript
# - For summaries, be comprehensive but organized

# Answer:"""
    
#     try:
#         # FIXED - NEW OPENAI SYNTAX
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             max_tokens=600 if is_summary_request else 400,
#             temperature=0.3
#         )
#         answer = response.choices[0].message.content.strip()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")
    
#     return {
#         "answer": answer,
#         "retrieved_chunks": len(indices[0]),
#         "relevance_scores": distances[0].tolist()[:3]  # Top 3 scores
#     }


# @app.delete("/delete/{video_id}")
# async def delete_video(video_id: str):
#     """Delete a video's index"""
#     vdir = VECTOR_DIR / video_id
    
#     if not vdir.exists():
#         raise HTTPException(status_code=404, detail="Video not found")
    
#     import shutil
#     shutil.rmtree(vdir)
    
#     return {"status": "success", "message": f"Deleted index for video {video_id}"}


# @app.get("/stats/{video_id}")
# async def get_video_stats(video_id: str):
#     """Get statistics about an indexed video"""
#     vdir = VECTOR_DIR / video_id
    
#     # Check if all required files exist
#     if not vdir.exists() or not (vdir / "index.faiss").exists():
#         raise HTTPException(
#             status_code=404, 
#             detail=f"Video '{video_id}' not indexed. Please ingest it first using /ingest endpoint."
#         )
    
#     metadata_file = vdir / "metadata.json"
#     chunks_file = vdir / "chunks.json"
    
#     if not metadata_file.exists() or not chunks_file.exists():
#         raise HTTPException(
#             status_code=500,
#             detail="Video index is incomplete. Please re-index using /ingest endpoint."
#         )
    
#     try:
#         with open(metadata_file, "r", encoding="utf-8") as f:
#             metadata = json.load(f)
        
#         with open(chunks_file, "r", encoding="utf-8") as f:
#             chunks = json.load(f)
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to read video data: {str(e)}"
#         )
    
#     total_words = sum(len(chunk.split()) for chunk in chunks)
    
#     return {
#         "video_id": video_id,
#         "total_chunks": len(chunks),
#         "total_words": total_words,
#         "duration": f"{metadata[-1]['end']:.0f}s",
#         "duration_formatted": format_timestamp(metadata[-1]['end']),
#         "avg_chunk_size": total_words // len(chunks) if chunks else 0
#     }


# if __name__ == "__main__":
#     import uvicorn
#     print("\n" + "="*60)
#     print("🚀 YouTube Transcript RAG - Production Backend")
#     print("="*60)
#     print(f"📍 Server: http://localhost:8000")
#     print(f"📚 Docs: http://localhost:8000/docs")
#     print(f"✨ Features: Timestamps, Smart chunking, Better prompts")
#     print("="*60 + "\n")
#     uvicorn.run(app, host="127.0.0.1", port=8000)




// // content_script.js
// // YouTube Summarizer Overlay with proper event handling

// const BACKEND_URL = "http://localhost:8000";

// if (!window.__YT_SUMMARIZER_INJECTED__) {
//   window.__YT_SUMMARIZER_INJECTED__ = true;
//   initSummarizerOverlay();
// }

// function initSummarizerOverlay() {
//   const container = document.createElement('div');
//   container.id = 'yt-summarizer-root';
//   container.style.all = 'initial';
//   document.documentElement.appendChild(container);
//   const shadow = container.attachShadow({ mode: 'open' });

//   // =========================
//   // Styles for the overlay UI
//   // =========================
//   const style = document.createElement('style');
//   style.textContent = `
//     * {
//       box-sizing: border-box;
//     }
//     .float-btn {
//       position: fixed;
//       right: 24px;
//       bottom: 24px;
//       width: 56px;
//       height: 56px;
//       border-radius: 50%;
//       box-shadow: 0 6px 20px rgba(0,0,0,0.25);
//       background: linear-gradient(135deg, #4f46e5, #06b6d4);
//       color: white;
//       display: flex;
//       align-items: center;
//       justify-content: center;
//       font-weight: bold;
//       font-size: 22px;
//       cursor: pointer;
//       z-index: 2147483647;
//       transition: transform 0.2s;
//     }
//     .float-btn:hover {
//       transform: scale(1.05);
//     }
//     .badge {
//       position: absolute;
//       top: -6px;
//       right: -6px;
//       background: #ff4757;
//       color: white;
//       font-size: 11px;
//       padding: 2px 6px;
//       border-radius: 999px;
//       font-weight: 600;
//     }
//     .overlay {
//       position: fixed;
//       right: 24px;
//       bottom: 92px;
//       width: 420px;
//       height: 520px;
//       background: #ffffff;
//       border-radius: 16px;
//       box-shadow: 0 12px 40px rgba(0,0,0,0.4);
//       display: flex;
//       flex-direction: column;
//       overflow: hidden;
//       z-index: 2147483647;
//       font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Arial;
//     }
//     .header {
//       display: flex;
//       align-items: center;
//       justify-content: space-between;
//       background: linear-gradient(90deg, #4f46e5, #06b6d4);
//       color: white;
//       padding: 12px 16px;
//     }
//     .header h3 {
//       margin: 0;
//       font-size: 15px;
//       font-weight: 600;
//     }
//     .actions {
//       display: flex;
//       gap: 8px;
//     }
//     .btn-ghost {
//       background: transparent;
//       border: 1px solid rgba(255,255,255,0.4);
//       color: white;
//       font-size: 11px;
//       padding: 5px 10px;
//       border-radius: 6px;
//       cursor: pointer;
//       transition: background 0.2s;
//       font-weight: 500;
//     }
//     .btn-ghost:hover {
//       background: rgba(255,255,255,0.1);
//     }
//     .status {
//       padding: 8px 12px;
//       background: #f0f9ff;
//       border-bottom: 1px solid #e0e7ff;
//       font-size: 12px;
//       color: #1e40af;
//     }
//     .chat {
//       flex: 1;
//       background: #f9fafb;
//       overflow-y: auto;
//       padding: 12px;
//     }
//     .msg {
//       margin: 8px 0;
//       display: flex;
//       animation: slideIn 0.2s ease-out;
//     }
//     @keyframes slideIn {
//       from { opacity: 0; transform: translateY(10px); }
//       to { opacity: 1; transform: translateY(0); }
//     }
//     .msg.user {
//       justify-content: flex-end;
//     }
//     .bubble {
//       padding: 10px 12px;
//       border-radius: 12px;
//       max-width: 80%;
//       word-wrap: break-word;
//       font-size: 14px;
//       line-height: 1.5;
//     }
//     .msg.user .bubble {
//       background: #e6fffa;
//       border-bottom-right-radius: 4px;
//     }
//     .msg.bot .bubble {
//       background: #f1f5f9;
//       border-bottom-left-radius: 4px;
//     }
//     .msg.bot .bubble.thinking {
//       background: #fef3c7;
//       font-style: italic;
//       color: #78350f;
//     }
//     .controls {
//       display: flex;
//       border-top: 1px solid #e5e7eb;
//       padding: 12px;
//       gap: 8px;
//       background: white;
//     }
//     .controls input {
//       flex: 1;
//       padding: 10px 12px;
//       border-radius: 8px;
//       border: 1px solid #d1d5db;
//       font-size: 14px;
//       outline: none;
//       font-family: inherit;
//     }
//     .controls input:focus {
//       border-color: #4f46e5;
//       box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
//     }
//     .controls button {
//       background: #4f46e5;
//       border: none;
//       color: white;
//       border-radius: 8px;
//       padding: 10px 16px;
//       cursor: pointer;
//       font-weight: 600;
//       transition: background 0.2s;
//     }
//     .controls button:hover {
//       background: #4338ca;
//     }
//     .controls button:disabled {
//       background: #9ca3af;
//       cursor: not-allowed;
//     }
//   `;
//   shadow.appendChild(style);

//   // =========================
//   // Floating button
//   // =========================
//   const floatBtn = document.createElement('div');
//   floatBtn.className = 'float-btn';
//   floatBtn.textContent = 'Y';
//   const badge = document.createElement('div');
//   badge.className = 'badge';
//   badge.textContent = 'AI';
//   floatBtn.appendChild(badge);
//   shadow.appendChild(floatBtn);

//   // =========================
//   // Chat overlay container
//   // =========================
//   const overlay = document.createElement('div');
//   overlay.className = 'overlay';
//   overlay.style.display = 'none';
//   overlay.innerHTML = `
//     <div class="header">
//       <h3>YouTube Summarizer</h3>
//       <div class="actions">
//         <button class="btn-ghost" id="summaryBtn">Summary</button>
//         <button class="btn-ghost" id="reindexBtn">Re-index</button>
//         <button class="btn-ghost" id="closeBtn">✕</button>
//       </div>
//     </div>
//     <div class="status" id="statusBar">Ready to answer questions</div>
//     <div class="chat" id="chatArea"></div>
//     <div class="controls">
//       <input id="queryInput" placeholder="Ask something about this video..." />
//       <button id="sendBtn">Send</button>
//     </div>
//   `;
//   shadow.appendChild(overlay);

//   // =========================
//   // Elements
//   // =========================
//   const chat = overlay.querySelector('#chatArea');
//   const sendBtn = overlay.querySelector('#sendBtn');
//   const queryInput = overlay.querySelector('#queryInput');
//   const closeBtn = overlay.querySelector('#closeBtn');
//   const summaryBtn = overlay.querySelector('#summaryBtn');
//   const reindexBtn = overlay.querySelector('#reindexBtn');
//   const statusBar = overlay.querySelector('#statusBar');

//   // =========================
//   // State
//   // =========================
//   let currentVideoId = null;
//   let isIndexed = false;
//   let isProcessing = false;

//   // =========================
//   // CRITICAL FIX: Stop event propagation
//   // This prevents YouTube keyboard shortcuts from triggering
//   // =========================
//   queryInput.addEventListener('keydown', (e) => {
//     e.stopPropagation();
//   });

//   queryInput.addEventListener('keyup', (e) => {
//     e.stopPropagation();
//   });

//   queryInput.addEventListener('keypress', (e) => {
//     e.stopPropagation();
//   });

//   // Also prevent other events
//   overlay.addEventListener('mousedown', (e) => {
//     e.stopPropagation();
//   });

//   overlay.addEventListener('click', (e) => {
//     e.stopPropagation();
//   });

//   // =========================
//   // Helper: Get current video ID
//   // =========================
//   function getCurrentVideoId() {
//     const urlParams = new URLSearchParams(window.location.search);
//     return urlParams.get('v');
//   }

//   // =========================
//   // Helper: Update status
//   // =========================
//   function updateStatus(message, type = 'info') {
//     statusBar.textContent = message;
//     statusBar.style.background = type === 'error' ? '#fee2e2' : 
//                                   type === 'success' ? '#d1fae5' : '#f0f9ff';
//     statusBar.style.color = type === 'error' ? '#991b1b' : 
//                            type === 'success' ? '#065f46' : '#1e40af';
//   }

//   // =========================
//   // Helper: Append message
//   // =========================
//   function appendMessage(text, who = 'bot', isThinking = false) {
//     const msg = document.createElement('div');
//     msg.className = `msg ${who}`;
//     const bubble = document.createElement('div');
//     bubble.className = isThinking ? 'bubble thinking' : 'bubble';
//     bubble.textContent = text;
//     msg.appendChild(bubble);
//     chat.appendChild(msg);
//     chat.scrollTop = chat.scrollHeight;
//     return bubble;
//   }

//   // =========================
//   // Helper: Remove last bot message
//   // =========================
//   function removeLastBotMessage() {
//     const botMessages = chat.querySelectorAll('.msg.bot');
//     if (botMessages.length > 0) {
//       botMessages[botMessages.length - 1].remove();
//     }
//   }

//   // =========================
//   // Function: Ingest video
//   // =========================
//   async function ingestVideo(videoId) {
//     updateStatus('Indexing transcript...', 'info');
//     try {
//       const response = await fetch(`${BACKEND_URL}/ingest`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ video_id: videoId })
//       });

//       const data = await response.json();

//       if (response.ok) {
//         isIndexed = true;
//         updateStatus(`✓ Indexed ${data.chunks || 0} chunks`, 'success');
//         return true;
//       } else {
//         updateStatus('Failed to index transcript', 'error');
//         appendMessage(`Error: ${data.detail || 'Failed to index'}`, 'bot');
//         return false;
//       }
//     } catch (e) {
//       updateStatus('Backend connection error', 'error');
//       appendMessage(`Connection error: ${e.message}`, 'bot');
//       return false;
//     }
//   }

//   // =========================
//   // Function: Query backend
//   // =========================
//   async function queryBackend(question) {
//     if (!currentVideoId) {
//       appendMessage('No video detected. Please open a YouTube video.', 'bot');
//       return;
//     }

//     if (isProcessing) return;

//     isProcessing = true;
//     sendBtn.disabled = true;
//     queryInput.disabled = true;

//     // Add user message
//     appendMessage(question, 'user');

//     // Add thinking indicator
//     const thinkingBubble = appendMessage('Thinking...', 'bot', true);

//     try {
//       // Ensure video is indexed
//       if (!isIndexed) {
//         const indexed = await ingestVideo(currentVideoId);
//         if (!indexed) {
//           removeLastBotMessage();
//           isProcessing = false;
//           sendBtn.disabled = false;
//           queryInput.disabled = false;
//           return;
//         }
//       }

//       // Query the backend
//       const response = await fetch(`${BACKEND_URL}/query`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ 
//           video_id: currentVideoId, 
//           question: question 
//         })
//       });

//       const data = await response.json();

//       // Remove thinking message
//       removeLastBotMessage();

//       if (response.ok) {
//         appendMessage(data.answer, 'bot');
//         updateStatus('Ready', 'success');
//       } else {
//         appendMessage(`Error: ${data.detail || 'Query failed'}`, 'bot');
//         updateStatus('Query failed', 'error');
//       }
//     } catch (e) {
//       removeLastBotMessage();
//       appendMessage(`Error: ${e.message}`, 'bot');
//       updateStatus('Connection error', 'error');
//     } finally {
//       isProcessing = false;
//       sendBtn.disabled = false;
//       queryInput.disabled = false;
//       queryInput.focus();
//     }
//   }

//   // =========================
//   // Function: Generate summary
//   // =========================
//   async function generateSummary() {
//     await queryBackend('Please provide a comprehensive summary of this video, including the main topics, key points, and important takeaways.');
//   }

//   // =========================
//   // Event: Toggle overlay
//   // =========================
//   floatBtn.addEventListener('click', () => {
//     const isVisible = overlay.style.display === 'flex';
//     overlay.style.display = isVisible ? 'none' : 'flex';
    
//     if (!isVisible) {
//       // Opening overlay
//       const vid = getCurrentVideoId();
//       if (vid !== currentVideoId) {
//         currentVideoId = vid;
//         isIndexed = false;
//         chat.innerHTML = '';
        
//         if (vid) {
//           updateStatus(`Video detected: ${vid}`, 'info');
//           appendMessage('👋 Hi! Ask me anything about this video.', 'bot');
//         } else {
//           updateStatus('No video detected', 'error');
//           appendMessage('Please open a YouTube video page.', 'bot');
//         }
//       }
//       queryInput.focus();
//     }
//   });

//   // =========================
//   // Event: Close overlay
//   // =========================
//   closeBtn.addEventListener('click', () => {
//     overlay.style.display = 'none';
//   });

//   // =========================
//   // Event: Send query
//   // =========================
//   sendBtn.addEventListener('click', () => {
//     const question = queryInput.value.trim();
//     if (!question || isProcessing) return;
    
//     queryBackend(question);
//     queryInput.value = '';
//   });

//   // =========================
//   // Event: Enter key to send
//   // =========================
//   queryInput.addEventListener('keydown', (e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       e.stopPropagation();
//       sendBtn.click();
//     }
//   });

//   // =========================
//   // Event: Summary button
//   // =========================
//   summaryBtn.addEventListener('click', () => {
//     if (!isProcessing) {
//       generateSummary();
//     }
//   });

//   // =========================
//   // Event: Re-index button
//   // =========================
//   reindexBtn.addEventListener('click', async () => {
//     if (isProcessing) return;
    
//     if (!currentVideoId) {
//       appendMessage('No video detected.', 'bot');
//       return;
//     }

//     isIndexed = false;
//     appendMessage('Re-indexing transcript...', 'bot');
//     const success = await ingestVideo(currentVideoId);
    
//     if (success) {
//       appendMessage('✓ Re-indexing complete! You can ask questions now.', 'bot');
//     }
//   });

//   // =========================
//   // Detect video changes
//   // =========================
//   let lastUrl = location.href;
//   new MutationObserver(() => {
//     const currentUrl = location.href;
//     if (currentUrl !== lastUrl) {
//       lastUrl = currentUrl;
//       const newVid = getCurrentVideoId();
      
//       if (newVid && newVid !== currentVideoId) {
//         currentVideoId = newVid;
//         isIndexed = false;
        
//         if (overlay.style.display === 'flex') {
//           chat.innerHTML = '';
//           updateStatus(`New video detected: ${newVid}`, 'info');
//           appendMessage('👋 New video detected! Ask me anything.', 'bot');
//         }
//       }
//     }
//   }).observe(document, { subtree: true, childList: true });
// }




// content_script.js
// YouTube Summarizer Overlay - Production Version with Beautiful Formatting
