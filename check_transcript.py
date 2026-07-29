from transcript_fetcher import TranscriptFetcher

# Initialize
fetcher = TranscriptFetcher()

# Get transcript
url = "https://www.youtube.com/watch?v=RFIUpNHsquE"
video_id = fetcher.extract_video_id(url)

# Get transcript
result = fetcher.get_transcript(video_id)
if result['success']:
    print(f"Got {result['total_segments']} segments")
    print(f"Duration: {fetcher.format_timestamp(result['duration'])}")
    print(f"Text: {result['full_text'][:500]}...")
else:
    print(f"Error: {result['error']}")
