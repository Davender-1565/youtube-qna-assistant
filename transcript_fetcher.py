"""
YouTube Transcript Fetcher
Professional implementation for fetching and processing YouTube video transcripts
Requires: youtube-transcript-api==1.2.4
"""
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict, Optional
import re


class TranscriptFetcher:
    """
    Fetch and process YouTube video transcripts

    Usage:
        fetcher = TranscriptFetcher()
        result = fetcher.get_transcript("dQw4w9WgXcQ")
        if result['success']:
            print(result['full_text'])
    """

    def __init__(self, cookies: str = None):
        """
        Initialize YouTube Transcript API

        Args:
            cookies: Not used (kept for compatibility)
        """
        self.api = YouTubeTranscriptApi()
        # cookies parameter kept for compatibility but not used,
        # library version doesn't support cookies

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats

        Supports:
            - https://www.youtube.com/watch?v=VIDEO_ID
            - https://youtu.be/VIDEO_ID
            - https://www.youtube.com/embed/VIDEO_ID
            - VIDEO_ID (plain ID)
        """
        if re.match(r'^[0-9A-Za-z_-]{11}$', url):
            return url

        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def get_transcript(
        self,
        video_id: str,
        languages: List[str] = ['en', 'en-US', 'en-GB']
    ) -> Dict:
        """
        Fetch transcript for a YouTube video

        Returns a dict with success, video_id, transcript, full_text,
        total_segments, duration, error.
        """
        try:
            data = self.api.fetch(video_id=video_id, languages=languages)

            transcript_list = []
            for segment in data:
                transcript_list.append({
                    "text": segment.text,
                    "start": segment.start,
                    "duration": segment.duration
                })

            full_text = " ".join([entry['text'] for entry in transcript_list])

            if transcript_list:
                last = transcript_list[-1]
                total_duration = last['start'] + last['duration']
            else:
                total_duration = 0

            return {
                "success": True,
                "video_id": video_id,
                "transcript": transcript_list,
                "full_text": full_text,
                "total_segments": len(transcript_list),
                "duration": total_duration,
                "error": None
            }

        except Exception as e:
            error_msg = str(e)

            if 'IpBlocked' in error_msg or 'Too Many Requests' in error_msg:
                error_msg = (
                    "YouTube blocked this request (IP blocking). "
                    "This happens on cloud servers. "
                    "Recommended solution: Fetch transcript in Chrome extension instead of backend."
                )

            return {
                "success": False,
                "video_id": video_id,
                "transcript": None,
                "full_text": None,
                "total_segments": 0,
                "duration": 0,
                "error": f"{type(e).__name__}: {error_msg}"
            }

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def search_in_transcript(
        self,
        transcript: List[Dict],
        query: str,
        case_sensitive: bool = False
    ) -> List[Dict]:
        """Search for segments containing specific text"""
        results = []
        search_query = query if case_sensitive else query.lower()

        for segment in transcript:
            text = segment['text'] if case_sensitive else segment['text'].lower()

            if search_query in text:
                results.append({
                    "text": segment['text'],
                    "start": segment['start'],
                    "duration": segment['duration'],
                    "timestamp": self.format_timestamp(segment['start'])
                })

        return results
