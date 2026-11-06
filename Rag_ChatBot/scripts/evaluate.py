"""
evaluate.py  -  A simple way to measure how good the bot is.

It checks three things:
  1. Retrieval hit rate : is the right FAQ inside the top-k results?
  2. Answer check       : does the answer contain the expected keyword?
  3. Fallback accuracy  : for questions outside our topic, does the bot say
                          "I don't know" instead of making something up?

Run:  python -m scripts.evaluate
(For deeper checks you can also turn on LangSmith tracing and look at each run.)
"""
import json
from pathlib import Path

from app import config
from app.database import init_db
from app.rag_pipeline import answer_question, retrieve

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"


def main() -> None:
    init_db()
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    in_scope, out_scope = data["in_scope"], data["out_of_scope"]

    hits, keyword_ok = 0, 0
    print("\n--- In-scope questions ---")
    for i, item in enumerate(in_scope):
        results = retrieve(item["question"], apply_threshold=False)
        found = any(doc.metadata.get("faq_id") == item["expected_faq_id"] for doc, _ in results)
        top_score = round(float(results[0][1]), 2) if results else 0.0

        reply = answer_question(f"eval-in-{i}", item["question"])
        has_keyword = item["keyword"].lower() in reply["answer"].lower()

        hits += found
        keyword_ok += has_keyword
        print(f"[{'OK' if found else 'MISS'}] retrieval | [{'OK' if has_keyword else 'MISS'}] answer | "
              f"top score {top_score} | {item['question']}")

    fallback_ok = 0
    print("\n--- Out-of-scope questions (bot should refuse politely) ---")
    for i, question in enumerate(out_scope):
        reply = answer_question(f"eval-out-{i}", question)
        fallback_ok += reply["used_fallback"]
        print(f"[{'OK' if reply['used_fallback'] else 'MISS'}] {question}")

    print("\n===== RESULTS =====")
    print(f"Settings            : top_k={config.TOP_K}, threshold={config.SIMILARITY_THRESHOLD}")
    print(f"Retrieval hit rate  : {hits}/{len(in_scope)} = {hits / len(in_scope):.0%}")
    print(f"Answer keyword rate : {keyword_ok}/{len(in_scope)} = {keyword_ok / len(in_scope):.0%}")
    print(f"Fallback accuracy   : {fallback_ok}/{len(out_scope)} = {fallback_ok / len(out_scope):.0%}")


if __name__ == "__main__":
    main()
