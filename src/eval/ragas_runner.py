import sys
import types

# Stub out VertexAI import that ragas tries to load
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub

import random
import argparse
from pathlib import Path
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

from src.eval.golden_set import load_golden_set, EvalQuestion
from src.ingest.parser import parse_directory
from src.ingest.chunker import chunk_sections
from src.index.bm25_store import build_bm25_retriever
from src.index.vector_store import build_vector_store, get_dense_retriever
from src.retrieve.hybrid import hybrid_retrieve
from src.retrieve.rerank import build_reranker, rerank
from src.generate.llm import build_llm, generate_answer

DATA_DIR = Path("data/processed/raw_md")
MODEL_PATH = "models/qwen2.5-7b-instruct-q3_k_m.gguf"


def run_pipeline_for_question(
    question: str,
    bm25_retriever,
    dense_retriever,
    reranker,
    llm,
) -> tuple[str, list[str]]:
    hybrid_results = hybrid_retrieve(question, bm25_retriever, dense_retriever, k=10)
    reranked_docs = rerank(question, hybrid_results, reranker, top_k=5)

    contexts = [doc.page_content for doc in reranked_docs]
    answer = generate_answer(question, reranked_docs, llm)

    return answer, contexts


def build_eval_dataset(
    questions: list[EvalQuestion],
    bm25_retriever,
    dense_retriever,
    reranker,
    llm,
) -> Dataset:
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q.question[:60]}...")
        answer, contexts = run_pipeline_for_question(
            q.question, bm25_retriever, dense_retriever, reranker, llm
        )
        data["question"].append(q.question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(q.ground_truth)

    return Dataset.from_dict(data)


def run_evaluation(num_questions: int = 10, seed: int = 42):
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== Ragas Evaluation ===\n")

    print("[1/5] Loading golden set...")
    all_questions = load_golden_set()
    random.seed(seed)
    selected = random.sample(all_questions, min(num_questions, len(all_questions)))
    print(f"       Selected {len(selected)} questions (seed={seed})")
    for q in selected:
        print(f"       [{q.difficulty}] {q.question[:50]}...")

    print("\n[2/5] Building pipeline...")
    sections = parse_directory(DATA_DIR)
    chunks = chunk_sections(sections)
    bm25_retriever = build_bm25_retriever(chunks, k=15)
    vector_store = build_vector_store(chunks)
    dense_retriever = get_dense_retriever(vector_store, k=15)
    reranker = build_reranker()

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,
        temperature=0.1,
        max_tokens=512,
        verbose=False,
    )

    print("\n[3/5] Running pipeline on selected questions...")
    eval_dataset = build_eval_dataset(
        selected, bm25_retriever, dense_retriever, reranker, llm
    )

    print("\n[4/5] Running Ragas evaluation (local LLM judge)...")
    judge_llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,
        temperature=0.0,
        max_tokens=1024,
        verbose=False,
    )

    judge_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
    )

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n[5/5] Results:\n")
    print(f"  Context Precision:  {results['context_precision']:.4f}")
    print(f"  Context Recall:     {results['context_recall']:.4f}")
    print(f"  Faithfulness:       {results['faithfulness']:.4f}")
    print(f"  Answer Relevancy:   {results['answer_relevancy']:.4f}")

    df = results.to_pandas()
    output_path = Path("src/eval/results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Per-question results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ragas evaluation")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(num_questions=args.num_questions, seed=args.seed)
