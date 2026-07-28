"""
LLM Handler with Conversation Memory
Manages OpenAI API calls and conversation history
"""
from openai import OpenAI
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class ConversationMemory:
    """Simple conversation memory storage"""

    def __init__(self, max_history: int = 5):
        """
        Initialize memory
        Args:
            max_history: Maximum Q&A pairs to remember (default: 5)
        """
        self.history = []
        self.max_history = max_history

    def add_exchange(
        self,
        question: str,
        answer: str,
        video_id: str,
        chunks_used: Optional[List[Dict]] = None
    ):
        """Add Q&A exchange to memory"""
        self.history.append({
            "question": question,
            "answer": answer,
            "video_id": video_id,
            "chunks": chunks_used or [],
            "timestamp": datetime.now().isoformat()
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_conversation_context(self, video_id: str, max_exchanges: int = 3) -> str:
        """Get formatted conversation history"""
        video_history = [h for h in self.history if h['video_id'] == video_id]
        if not video_history:
            return ""
        recent = video_history[-max_exchanges:] if len(video_history) > max_exchanges else video_history
        context = "Previous conversation in this video:\n\n"
        for i, exchange in enumerate(recent, 1):
            context += f"Q{i}: {exchange['question']}\n"
            answer_preview = exchange['answer'][:150] + "..." if len(exchange['answer']) > 150 else exchange['answer']
            context += f"A{i}: {answer_preview}\n\n"
        return context

    def clear(self):
        """Clear all memory"""
        self.history = []

    def clear_video(self, video_id: str):
        """Clear memory for specific video"""
        self.history = [h for h in self.history if h['video_id'] != video_id]

    def get_summary(self, video_id: str) -> Dict:
        """Get learning summary"""
        video_history = [h for h in self.history if h['video_id'] == video_id]
        return {
            "total_questions": len(video_history),
            "questions": [h['question'] for h in video_history],
            "last_question": video_history[-1]['question'] if video_history else None,
            "last_timestamp": video_history[-1]['timestamp'] if video_history else None
        }


class LLMHandler:
    """LLM Handler with Memory Support"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """Initialize LLM handler"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = ConversationMemory()
        print(f"  LLM Handler initialized")
        print(f"   Model: {model}")
        print(f"   Memory: Enabled")

    def answer_question(
        self,
        question: str,
        context: str,
        video_id: str,
        use_memory: bool = True
    ) -> Dict:
        """Generate answer using RAG context and memory"""
        conversation_history = ""
        if use_memory:
            conversation_history = self.memory.get_conversation_context(video_id)

        system_prompt = """You are a helpful AI assistant that answers questions about YouTube videos.
You will be given:
1. A user's question
2. Relevant excerpts from the video transcript with timestamps
3. Previous conversation history (if any)
Your job:
- Answer the question accurately using the provided context
- Use conversation history to understand follow-up questions
- Include specific timestamp references (format: [MM:SS])
- Be concise but complete
- If context doesn't have the answer, say so honestly
- For follow-up questions, reference previous answers when helpful
Remember: Only use information from the provided transcript context."""

        user_message = f"""Question: {question}
{conversation_history}
Video Transcript Context:
{context}
Please answer the question based on the transcript context above."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            answer = response.choices[0].message.content
            usage = response.usage

            if use_memory:
                self.memory.add_exchange(
                    question=question,
                    answer=answer,
                    video_id=video_id
                )

            return {
                "success": True,
                "answer": answer,
                "question": question,
                "video_id": video_id,
                "model": self.model,
                "tokens_used": {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens
                },
                "memory_enabled": use_memory,
                "conversation_length": len(self.memory.history),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "question": question,
                "video_id": video_id,
                "error": f"{type(e).__name__}: {str(e)}"
            }

    def get_learning_summary(self, video_id: str) -> str:
        """Get learning summary"""
        summary = self.memory.get_summary(video_id)
        if summary['total_questions'] == 0:
            return "You haven't asked any questions about this video yet."
        response = f" Learning Summary for this video:\n\n"
        response += f"Total questions asked: {summary['total_questions']}\n\n"
        response += f"Topics explored:\n"
        for i, question in enumerate(summary['questions'], 1):
            response += f"{i}. {question}\n"
        return response

    def clear_memory(self, video_id: Optional[str] = None):
        """Clear conversation memory"""
        if video_id:
            self.memory.clear_video(video_id)
            print(f" Memory cleared for video: {video_id}")
        else:
            self.memory.clear()
            print(f" All memory cleared")
