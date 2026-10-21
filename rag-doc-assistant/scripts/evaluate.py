"""Retrieval hit-rate evaluation.

1. Upload your document through the app first.
2. Edit tests.json-style list below: question + the page where the answer lives.
3. Run from project root:  python -m scripts.evaluate
"""
from app.rag import retrieve

# EDIT THESE for your own document
EVAL_SET = [
    {"question": "What is the main topic of the document?", "file": "sample.pdf", "page": 1},
    # add 10+ more...
]


def main():
    hits = 0
    for item in EVAL_SET:
        results = retrieve(item["question"])
        found = any(r["source"] == item["file"] and r["page"] == item["page"] for r in results)
        hits += found
        print(("HIT  " if found else "MISS ") + item["question"])
    print(f"\nRetrieval hit-rate: {hits}/{len(EVAL_SET)} = {hits/len(EVAL_SET):.0%}")


if __name__ == "__main__":
    main()
