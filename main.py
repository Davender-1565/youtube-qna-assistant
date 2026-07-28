"""
YouTube Learning Assistant - FastAPI Backend
Complete API server integrating RAG + LLM + Memory
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from dotenv import load_dotenv
import os
from datetime import datetime

from transcript_fetcher import TranscriptFetcher
from rag_system import RAGSystem
from llm_handler import LLMHandler

load_dotenv()

app = FastAPI(
    title="YouTube Learning Assistant API",
    description="AI-powered YouTube video Q&A assistant with hybrid RAG",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your extension origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

youtube_cookies = os.getenv("YOUTUBE_COOKIES")
transcript_fetcher = TranscriptFetcher(cookies=youtube_cookies)
llm_handler = LLMHandler(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("MAX_TOKENS", "500"))
)

video_sessions: Dict[str, RAGSystem] = {}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class VideoQueryRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID", min_length=11, max_length=11)
    question: str = Field(..., description="User's question", min_length=1)
    use_memory: bool = Field(True, description="Use conversation memory")
    strategy: str = Field("hybrid_rerank", description="Search strategy")

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "RFIUpNHsquE",
                "question": "What is this video about?",
                "use_memory": True,
                "strategy": "hybrid_rerank"
            }
        }


class VideoQueryResponse(BaseModel):
    """ response model for video query """ 
    success: bool
    video_id: str
    question: str
    answer: str
    context_used: Optional[int] = None
    tokens_used: Optional[int] = None
    conversation_length: Optional[int] = None
    strategy_used: Optional[str] = None
    error: Optional[str] = None


class TranscriptRequest(BaseModel):
    """Request model for transcript fetching"""
    video_id: str = Field(..., min_length=11, max_length=11)
    video_title: Optional[str] = None


class TranscriptResponse(BaseModel):
    """Response model for transcript"""
    success: bool
    video_id: str
    total_segments: int
    duration: float
    full_text: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    openai_configured: bool
    total_sessions: int
    environment: str
    timestamp: str


class ClearMemoryRequest(BaseModel):
    """Request to clear memory"""
    video_id: Optional[str] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
async def get_or_create_rag_session(video_id: str, video_title: str = "") -> RAGSystem:
    """Get existing RAG session or create new one
    
    Args:
        video_id: YouTube video ID
        video_title: Optional video title
        
    Returns:
        RAGSystem instance with built index
    """
    if video_id in video_sessions:
        print(f" Using cached RAG session for {video_id}")
        return video_sessions[video_id]

    print(f" Creating new RAG session for {video_id}")

    transcript_result = transcript_fetcher.get_transcript(video_id)
    if not transcript_result['success']:
        raise HTTPException(
            status_code=404,
            detail=f"Failed to fetch transcript: {transcript_result['error']}"
        )

    rag = RAGSystem()
    rag.build_index(
        video_id=video_id,
        transcript=transcript_result['transcript'],
        video_title=video_title
    )

    video_sessions[video_id] = rag
    print(f" RAG session created and cached for {video_id}")
    return rag


def cleanup_old_sessions(max_sessions: int = 10):
    """Keep only the most recent max_sessions"""
    if len(video_sessions) > max_sessions:
        sessions_to_remove = len(video_sessions) - max_sessions
        for video_id in list(video_sessions.keys())[:sessions_to_remove]:
            del video_sessions[video_id]
            print(f"🗑️  Cleaned up old session: {video_id}")


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    return {
        "status": "active",
        "message": "YouTube Learning Assistant API",
        "version": "2.0.0",
        "features": [
            "Hybrid RAG (Semantic + BM25)",
            "Cross-encoder re-ranking",
            "Conversation memory",
            "Multi-strategy search"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "query": "/api/query",
            "transcript": "/api/transcript",
            "memory": "/api/memory"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        openai_configured=bool(os.getenv("OPENAI_API_KEY")),
        total_sessions=len(video_sessions),
        environment="development" if os.getenv("DEBUG") == "True" else "production",
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/query", response_model=VideoQueryResponse)
async def query_video(request: VideoQueryRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint: Query a YouTube video with a question
    
    Process:
    1. Get or create RAG session for video
    2. Search for relevant context
    3. Generate answer with LLM
    4. Return answer with metadata
    """
    try:
        print(f"\n{'='*70}")
        print(f" New Query")
        print(f"{'='*70}")
        print(f"Video ID: {request.video_id}")
        print(f"Question: {request.question}")
        print(f"Strategy: {request.strategy}")
        print(f"{'='*70}")

        rag = await get_or_create_rag_session(request.video_id)

        context_data = rag.get_context(
            query=request.question,
            top_k=3,
            max_tokens=1500,
            strategy=request.strategy
        )

        print(f"\n Context retrieved:")
        print(f"   Chunks: {context_data['num_chunks']}")
        print(f"   Strategy: {context_data['strategy']}")

        llm_response = llm_handler.answer_question(
            question=request.question,
            context=context_data['context'],
            video_id=request.video_id,
            use_memory=request.use_memory
        )

        if not llm_response['success']:
            raise HTTPException(
                status_code=500,
                detail=f"LLM error: {llm_response['error']}"
            )

        print(f"\n Answer generated:")
        print(f"   Tokens: {llm_response['tokens_used']['total']}")
        print(f"   Memory: {llm_response['conversation_length']} exchanges")
        print(f"{'='*70}\n")

        background_tasks.add_task(cleanup_old_sessions, max_sessions=10)

        return VideoQueryResponse(
            success=True,
            video_id=request.video_id,
            question=request.question,
            answer=llm_response['answer'],
            context_used=context_data['num_chunks'],
            tokens_used=llm_response['tokens_used']['total'],
            conversation_length=llm_response['conversation_length'],
            strategy_used=request.strategy
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f" Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcript", response_model=TranscriptResponse)
async def get_transcript(request: TranscriptRequest):
    """
    Fetch transcript for a video (without building RAG index)
    Useful for preview or verification
    """
    try:
        result = transcript_fetcher.get_transcript(request.video_id)
        if not result['success']:
            raise HTTPException(
                status_code=404,
                detail=f"Transcript not found: {result['error']}"
            )
        return TranscriptResponse(
            success=True,
            video_id=request.video_id,
            total_segments=result['total_segments'],
            duration=result['duration'],
            full_text=result['full_text'][:500] + "..." if len(result['full_text']) > 500 else result['full_text']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/create")
async def create_session(request: TranscriptRequest):
    try:
        rag = await get_or_create_rag_session(
            request.video_id,
            request.video_title or ""
        )
        stats = rag.get_stats()
        return {
            "success": True,
            "video_id": request.video_id,
            "message": "Session created and cached",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{video_id}/stats")
async def get_session_stats(video_id: str):
    if video_id not in video_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No session found for video: {video_id}"
        )
    rag = video_sessions[video_id]
    stats = rag.get_stats()
    return {
        "success": True,
        "video_id": video_id,
        "stats": stats
    }


@app.delete("/api/session/{video_id}")
async def delete_session(video_id: str):
    if video_id in video_sessions:
        del video_sessions[video_id]
        return {
            "success": True,
            "message": f"Session deleted for video: {video_id}"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No session found for video: {video_id}"
        )


@app.post("/api/memory/clear")
async def clear_memory(request: ClearMemoryRequest):
    try:
        llm_handler.clear_memory(request.video_id)
        return {
            "success": True,
            "message": f"Memory cleared for {'video: ' + request.video_id if request.video_id else 'all videos'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/{video_id}/summary")
async def get_memory_summary(video_id: str):
    try:
        summary = llm_handler.get_learning_summary(video_id)
        return {
            "success": True,
            "video_id": video_id,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    for video_id, rag in video_sessions.items():
        stats = rag.get_stats()
        sessions.append({
            "video_id": video_id,
            "video_title": stats.get('video_title', 'Unknown'),
            "chunks": stats.get('total_chunks', 0)
        })
    return {
        "success": True,
        "total_sessions": len(sessions),
        "sessions": sessions
    }


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("\n" + "="*70)
    print(" YouTube Learning Assistant API Starting...")
    print("="*70)
    print(f"Environment: {'Development' if os.getenv('DEBUG') == 'True' else 'Production'}")
    print(f"OpenAI Configured: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"Server: http://localhost:{os.getenv('PORT', '8000')}")
    print(f"Docs: http://localhost:{os.getenv('PORT', '8000')}/docs")
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "="*70)
    print(" Shutting down YouTube Learning Assistant API...")
    print(f"Cleaned up {len(video_sessions)} sessions")
    print("="*70 + "\n")
    video_sessions.clear()


# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Important: Must be 0.0.0.0 for Render
        port=port,
        reload=False  # Disable reload in production
    )
