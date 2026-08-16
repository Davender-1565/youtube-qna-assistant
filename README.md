# YouTube Q&A Assistant

AI-powered YouTube video Q&A assistant with **hybrid RAG** (Retrieval-Augmented Generation). Built with FastAPI, FAISS, BM25, cross-encoder re-ranking, and conversation memory.

## Features

- **Hybrid Search**: Combines semantic embeddings (768-dim) + BM25 keyword search
- **Cross-Encoder Re-ranking**: ms-marco-MiniLM-L-6-v2 for precision
- **Conversation Memory**: Remembers previous Q&A per video (5 exchanges)
- **Multiple Search Strategies**: semantic / bm25 / hybrid / hybrid_rerank (recommended)
- **Timestamp References**: Answers include `[MM:SS]` citations
- **Clickable Timestamps**: Click timestamps in chat to seek video to that moment
- **Session Management**: Cache RAG indexes for fast repeated queries
- **Chrome Extension Ready**: CORS enabled, RESTful API

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  YouTube    │────▶│  Transcript  │────▶│   Chunking  │
│  Video ID   │     │   Fetcher    │     │  (overlap)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   LLM       │◀────│   Context    │◀────│  Hybrid     │
│  (GPT-4o)   │     │  Assembly    │     │  RAG Index  │
└─────────────┘     └──────────────┘     └─────────────┘
                           ▲
                    ┌──────┴──────┐
                    │  Memory     │
                    │  (5 turns)  │
                    └─────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone and navigate
cd youtube-qna-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # Create .env with your OPENAI_API_KEY
```

### Environment Variables (`.env`)

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=500
PORT=8000
DEBUG=True
# YOUTUBE_COOKIES=optional  # For private videos
```

### Run Server

```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at `http://localhost:8000`  
API Docs: `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info & features |
| `GET` | `/health` | Health check |
| `POST` | `/api/query` | **Main: Ask a question about a video** |
| `POST` | `/api/transcript` | Fetch transcript only |
| `POST` | `/api/session/create` | Pre-create RAG session |
| `GET` | `/api/session/{video_id}/stats` | Get session stats |
| `DELETE` | `/api/session/{video_id}` | Delete cached session |
| `GET` | `/api/sessions` | List all active sessions |
| `POST` | `/api/memory/clear` | Clear conversation memory |
| `GET` | `/api/memory/{video_id}/summary` | Get learning summary |

### Main Query Endpoint

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "RFIUpNHsquE",
    "question": "What is this video about?",
    "use_memory": true,
    "strategy": "hybrid_rerank"
  }'
```

**Response:**
```json
{
  "success": true,
  "video_id": "RFIUpNHsquE",
  "question": "What is this video about?",
  "answer": "This video explains [topic]... [01:23] ... [04:56]",
  "context_used": 3,
  "tokens_used": 487,
  "conversation_length": 1,
  "strategy_used": "hybrid_rerank"
}
```

### Search Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `semantic` | Pure embedding similarity | Conceptual questions |
| `bm25` | Pure keyword matching | Exact terms, names, codes |
| `hybrid` | 50/50 semantic + BM25 | Balanced |
| `hybrid_rerank` | Hybrid + cross-encoder (recommended) | **Best accuracy** |

## Project Structure

```
youtube-qna-assistant/
├── main.py              # FastAPI server & endpoints
├── rag_system.py        # Hybrid RAG (FAISS + BM25 + CrossEncoder)
├── llm_handler.py       # OpenAI + ConversationMemory
├── transcript_fetcher.py# YouTube transcript fetching
├── check_transcript.py  # Quick transcript test script
├── test_rag.py          # RAG system tests
├── requirements.txt     # Dependencies
└── .env                 # Environment variables (create this)
```

## How It Works

1. **Transcript Fetching**: `youtube-transcript-api` gets captions (handles IP blocking)
2. **Chunking**: Overlapping 5-segment chunks with 2-segment overlap
3. **Indexing**: 
   - FAISS IndexFlatL2 (768-dim `all-mpnet-base-v2` embeddings)
   - BM25Okapi (tokenized chunks)
4. **Search**: Hybrid scoring (`alpha * semantic + (1-alpha) * BM25`)
5. **Re-ranking**: Cross-encoder `ms-marco-MiniLM-L-6-v2`
6. **Context**: Top-k chunks formatted with timestamps
7. **Generation**: GPT-4o-mini with system prompt + memory
8. **Memory**: Last 5 Q&A exchanges per video

## Deployment

### Render / Railway / Fly.io

```yaml
# render.yaml example
services:
  - type: web
    name: youtube-qna-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: PORT
        value: "8000"
```

**Important**: Use `host="0.0.0.0"` in `main.py` (already configured).

### Chrome Extension Integration

```javascript
// content.js / popup.js
const API_BASE = "https://your-api.onrender.com";

async function askQuestion(videoId, question) {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, question })
  });
  return res.json();
}
```

## Chrome Extension (Local Development)

A companion Chrome extension is available in `../youtube-chat-extension/`.

### Features
- **Side Panel UI**: Opens alongside YouTube video
- **Auto-detects Video**: Reads video ID from URL automatically
- **Conversation Memory**: Remembers 5 Q&A exchanges per video
- **Clickable Timestamps**: Click `[MM:SS]` in answers to seek video
- **Quick Suggestions**: Pre-built question buttons
- **Token/Strategy Display**: Shows tokens used & search strategy per answer
- **Dark Theme**: Matches YouTube's dark mode

### Setup

1. **Start the backend** (in `youtube-qna-assistant/`):
   ```bash
   python main.py
   ```

2. **Load extension in Chrome**:
   - Open `chrome://extensions/`
   - Enable **Developer mode** (top right toggle)
   - Click **Load unpacked**
   - Select the `youtube-chat-extension` folder

3. **Use it**:
   - Open any YouTube video (`youtube.com/watch?v=...`)
   - Click the extension icon in toolbar → Opens side panel
   - Or click the side panel button (puzzle piece → pin "YouTube AI Chat Assistant")
   - Ask questions about the video!

### Extension Files
```
youtube-chat-extension/
├── manifest.json        # Manifest V3 (sidePanel + content script + background)
├── background.js        # Service worker (API proxy to localhost:8000)
├── content.js           # Extracts video ID, seeks video on timestamp click
├── sidepanel.html       # Chat UI structure
├── sidepanel.js         # Chat logic (send msg, format timestamps, seek video)
├── sidepanel.css        # Dark theme styling
└── icons/               # SVG icons (16, 48, 128)
```

### Extension Architecture
```
┌─────────────────┐     HTTP/WS      ┌─────────────────┐
│  Chrome Ext     │ ◄─────────────►  │  FastAPI Backend│
│  (Sidepanel)    │   /api/query     │  (your code)    │
└────────┬────────┘                  └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Content Script │──► Extracts video_id, seeks video
│  (youtube.com)  │     via postMessage / video.currentTime
└─────────────────┘
```

## Testing

```bash
# Run RAG tests
python test_rag.py

# Quick transcript check
python check_transcript.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `IpBlocked` / `Too Many Requests` | Fetch transcript in Chrome extension (client-side), send text to API |
| `OPENAI_API_KEY not found` | Add to `.env` file |
| Slow first query | RAG index builds on-demand; pre-create with `/api/session/create` |
| Memory not working | Ensure `use_memory: true` in request |
| Extension input disabled | Refresh YouTube page, reload extension, check backend running on localhost:8000 |
| Timestamps not clickable | Reload extension & refresh YouTube page |
| Video doesn't seek on timestamp click | Check YouTube page console for "No video player found" - refresh page |
