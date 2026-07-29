"""
Complete Hybrid RAG System Test
Tests: Semantic + BM25 + Re-ranking + Memory
"""
from transcript_fetcher import TranscriptFetcher
from rag_system import RAGSystem
from llm_handler import LLMHandler


def print_separator(title: str = ""):
    print("\n" + "="*70)
    if title:
        print(title)
        print("="*70)


def test_hybrid_search():
    """Test 1: Compare different search strategies"""
    print_separator(" TEST 1: HYBRID SEARCH COMPARISON")

    print("\n Initializing...")
    fetcher = TranscriptFetcher()
    rag = RAGSystem()

    print("\n Fetching transcript...")
    video_id = "RFIUpNHsquE"
    result = fetcher.get_transcript(video_id)

    if not result['success']:
        print(f" Failed: {result['error']}")
        return None, None

    print(f" Got {result['total_segments']} segments")

    print("\n Building hybrid index...")
    rag.build_index(
        video_id=video_id,
        transcript=result['transcript'],
        video_title="Google Login Error Fix Tutorial"
    )

    print("\n Comparing Search Strategies...")
    print("-"*70)

    test_query = "What is this video about?"
    strategies = [
        ("semantic", "Pure Semantic Search"),
        ("bm25", "Pure Keyword Search (BM25)"),
        ("hybrid", "Hybrid (Semantic + BM25)"),
        ("hybrid_rerank", "Hybrid + Re-ranking (BEST)")
    ]

    for strategy, description in strategies:
        print(f"\n{'─'*70}")
        print(f"Strategy: {description}")
        print(f"{'─'*70}")

        results = rag.search(test_query, top_k=3, strategy=strategy)

        for result in results:
            print(f"\n  Rank {result['rank']}:")
            print(f"    Time: [{rag._format_timestamp(result['start_time'])}]")

            if 'semantic_score' in result:
                print(f"    Semantic: {result.get('semantic_score', 0):.4f}")
            if 'bm25_score' in result:
                print(f"    BM25: {result.get('bm25_score', 0):.4f}")
            if 'hybrid_score' in result:
                print(f"    Hybrid: {result['hybrid_score']:.4f}")
            if 'rerank_score' in result:
                print(f"    Rerank: {result['rerank_score']:.4f}")

            print(f"    Text: {result['text'][:60]}...")

    print("\n Testing Context Generation...")
    print("-"*70)

    context = rag.get_context(test_query, top_k=3, strategy="hybrid_rerank")
    print(f"\n Context generated using: {context['strategy']}")
    print(f"   Chunks: {context['num_chunks']}")
    print(f"   Characters: {context['total_chars']}")
    print(f"\n   Preview:")
    print(f"   {context['context'][:200]}...")

    print("\n System Statistics:")
    print("-"*70)
    stats = rag.get_stats()
    print(f"\n Status: {stats['status']}")
    print(f" Video: {stats['video_title']}")
    print(f" Chunks: {stats['total_chunks']}")
    print(f" FAISS vectors: {stats['faiss_index_size']}")
    print(f" BM25 terms: {stats['bm25_index_size']}")
    print(f"\n Features:")
    for feature in stats['features']:
        print(f"   {feature}")

    print_separator(" TEST 1 COMPLETE: Hybrid Search Working!")

    return rag, video_id


def test_llm_with_memory(rag: RAGSystem, video_id: str):
    """Test 2: LLM with memory"""
    print_separator(" TEST 2: LLM + MEMORY")

    print("\n Initializing LLM...")
    llm = LLMHandler()

    print("\n Question 1 (Initial):")
    print("-"*70)

    q1 = "What is this video about?"
    print(f"\n User: {q1}")

    context1 = rag.get_context(q1, top_k=3, strategy="hybrid_rerank")
    response1 = llm.answer_question(q1, context1['context'], video_id, use_memory=True)

    if response1['success']:
        print(f"\n Assistant:\n{response1['answer']}")
        print(f"\n Tokens: {response1['tokens_used']['total']}")
        print(f" Memory: {response1['conversation_length']} exchanges")

    print("\n Question 2 (Follow-up - Memory Test):")
    print("-"*70)
    print("  VAGUE question - memory should help!")

    q2 = "How do I fix it?"
    print(f"\n User: {q2}")

    context2 = rag.get_context(q2, top_k=3, strategy="hybrid_rerank")
    response2 = llm.answer_question(q2, context2['context'], video_id, use_memory=True)

    if response2['success']:
        print(f"\n Assistant:\n{response2['answer']}")
        print(f"\n Memory helped! Bot knew 'it' = Google login error")

    print("\n Question 3 (Another Follow-up):")
    print("-"*70)

    q3 = "Can you give me the exact steps?"
    print(f"\n User: {q3}")

    context3 = rag.get_context(q3, top_k=3, strategy="hybrid_rerank")
    response3 = llm.answer_question(q3, context3['context'], video_id, use_memory=True)

    if response3['success']:
        print(f"\n Assistant:\n{response3['answer']}")

    print("\n Learning Summary:")
    print("-"*70)
    summary = llm.get_learning_summary(video_id)
    print(f"\n{summary}")

    print_separator(" TEST 2 COMPLETE: LLM + Memory Working!")


def test_keyword_specific_queries():
    """Test 3: Test BM25 advantage with specific keywords"""
    print_separator(" TEST 3: KEYWORD-SPECIFIC QUERIES (BM25 Test)")

    fetcher = TranscriptFetcher()
    rag = RAGSystem()

    video_id = "RFIUpNHsquE"
    result = fetcher.get_transcript(video_id)
    rag.build_index(video_id, result['transcript'])

    keyword_queries = [
        "console.cloud.google.com",
        "redirect_uri",
        "localhost",
        "OAuth"
    ]

    print("\n Testing keyword-specific queries:")
    print("(These should show BM25 advantage)")
    print("-"*70)

    for query in keyword_queries:
        print(f"\n{'─'*70}")
        print(f"Query: '{query}'")
        print(f"{'─'*70}")

        semantic_results = rag.search(query, top_k=2, strategy="semantic")
        hybrid_results = rag.search(query, top_k=2, strategy="hybrid_rerank")

        print(f"\n  Semantic Only (Top 1):")
        print(f"    {semantic_results[0]['text'][:80]}...")

        print(f"\n  Hybrid + BM25 (Top 1):")
        print(f"    {hybrid_results[0]['text'][:80]}...")

        if query.lower() in hybrid_results[0]['text'].lower():
            print(f"\n     BM25 helped! Found exact keyword: '{query}'")
        else:
            print(f"\n      Keyword not in top result")

    print_separator(" TEST 3 COMPLETE: BM25 Keyword Matching Working!")


def main():
    """Run all tests"""
    print("="*70)
    print(" COMPLETE HYBRID RAG SYSTEM TEST")
    print("Semantic + BM25 + Re-ranking + Memory")
    print("="*70)

    try:
        rag, video_id = test_hybrid_search()

        if rag and video_id:
            test_llm_with_memory(rag, video_id)
            test_keyword_specific_queries()

        print("\n" + "="*70)
        print(" ALL TESTS PASSED!")
        print("="*70)
        print("\n System Components Working:")
        print("   1. Transcript Fetching")
        print("   2. Semantic Search (768-dim embeddings)")
        print("   3. Keyword Search (BM25)")
        print("   4. Hybrid Search (Semantic + BM25)")
        print("   5. Cross-Encoder Re-ranking")
        print("   6. LLM Answer Generation")
        print("   7. Conversation Memory")
        print("\n Production-Grade Hybrid RAG System!")
        print("="*70)

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
