from backend.retrieval.query_parser import QueryParser
from backend.retrieval.search import SemanticSearcher
from backend.retrieval.reranker import AttributeReranker

def main():
    # Keep LLM off for offline test (stable + no API key dependency)
    parser = QueryParser(llm_threshold=0.7)
    searcher = SemanticSearcher()
    reranker = AttributeReranker()

    queries = [
        "A person in a bright yellow raincoat.",
        "blue shirt black pants",
        "A red tie and a white shirt in a formal setting.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
        "A model walking on a runway wearing black outfit."
    ]

    for q in queries:
        parsed, conf, fallback = parser.parse(q, use_llm=True)

        print("\n" + "="*80)
        print("QUERY:", q)
        print("PARSED:", parsed, "| conf:", round(conf, 2), "| fallback_used:", fallback)

        # Retrieve more candidates first, then rerank to top_k
        candidates = searcher.search(q, parsed_attrs=parsed, top_k=50)
        final = reranker.rerank(candidates, parsed_attrs=parsed, top_k=8)

        print(f"Retrieved {len(candidates)} candidates -> Returning {len(final)} results\n")

        for i, r in enumerate(final, 1):
            print(
                f"{i:02d}. score={r.get('score', 0):.3f} "
                f"id={r.get('image_id')} "
                f"env={r.get('environment')} "
                f"cloth={r.get('clothing_type')} "
                f"vibe={r.get('vibe')} "
                f"colors={r.get('colors')}"
            )
            print("    path:", r.get("image_path"))
            print("    why :", r.get("explanation"))

if __name__ == "__main__":
    main()
