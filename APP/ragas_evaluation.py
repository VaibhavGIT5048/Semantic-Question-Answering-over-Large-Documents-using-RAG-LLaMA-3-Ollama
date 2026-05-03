import os
import argparse
import asyncio
import json
import random
import re
import pickle
import pandas as pd
import time
from openai import AsyncOpenAI
from ragas import experiment
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────────────────────────────
# 1. INITIALIZATION
# ─────────────────────────────────────────────────────────────────────
print("🚀 Initializing High-Performance Evaluation Engine...")

# Ollama OpenAI-compatible endpoint (best speed + structured eval support)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
STUDENT_MODEL = os.getenv("OLLAMA_STUDENT_MODEL", "llama3.1:8b")
JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", STUDENT_MODEL)
GENERATOR_MODEL = os.getenv("OLLAMA_GENERATOR_MODEL", STUDENT_MODEL)

# Student LLM (Lower temperature for Faithfulness)
llm = OllamaLLM(model=STUDENT_MODEL, temperature=0)

# Judge LLM (async-capable)
judge_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Indices
vs = FAISS.load_local("indexes/faiss_index", emb_model, allow_dangerous_deserialization=True)
with open("indexes/bm25_data.pkl", "rb") as f:
    bm_data = pickle.load(f)
    bm25, chunks_ref = bm_data["bm25"], bm_data["chunks"]

# ─────────────────────────────────────────────────────────────────────
# 2. JUDGE PROMPT (FAST + ROBUST PARSING)
# ─────────────────────────────────────────────────────────────────────
VERDICT_PATTERN = re.compile(r"\b(Excellent|Hallucinated|Irrelevant)\b", re.IGNORECASE)
REFUSAL_PATTERN = re.compile(
    r"\b(cannot find|not in the document|not provided in the document|not available in the document|cannot locate)\b",
    re.IGNORECASE,
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. Return a verdict using ONLY one of: "
    "Excellent, Hallucinated, Irrelevant."
)

def build_judge_prompt(question: str, response: str, contexts: list[str], ground_truth: str) -> str:
    return f"""Evaluate the response based on the Context and Ground Truth provided.

Context: {" ".join(contexts)}
Question: {question}
Response: {response}
Ground Truth: {ground_truth}

JUDGMENT CRITERIA:
1. FAITHFULNESS: Does the response contain ANY information NOT present in the Context? If yes, it is Hallucinated.
2. RELEVANCY: Does the response answer the Question directly without fluff?
3. ACCURACY: Does the response match the facts in the Ground Truth?

Return exactly two lines:
Verdict: <Excellent|Hallucinated|Irrelevant>
Reason: <short reason>
"""

async def judge_verdict(question: str, response: str, contexts: list[str], ground_truth: str) -> tuple[str, str]:
    prompt = build_judge_prompt(question, response, contexts, ground_truth)
    completion = await judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=200,
    )
    content = completion.choices[0].message.content or ""
    match = VERDICT_PATTERN.search(content)
    verdict = match.group(1).title() if match else "Irrelevant"
    return verdict, content.strip()

# ─────────────────────────────────────────────────────────────────────
# 3. DATASET GENERATION FROM CHUNKS
# ─────────────────────────────────────────────────────────────────────
QA_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

def load_chunks_jsonl(path: str) -> list[str]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            content = record.get("page_content", "")
            if content:
                chunks.append(content)
    return chunks

async def generate_qa_from_chunk(chunk_text: str) -> dict | None:
    prompt = f"""Generate ONE question and its ground-truth answer strictly from the context.
Return a JSON object with keys: question, ground_truth.

Context:
{chunk_text}

JSON:"""
    completion = await judge_client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": "You generate concise QA pairs from context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    content = completion.choices[0].message.content or ""
    match = QA_JSON_PATTERN.search(content)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not payload.get("question") or not payload.get("ground_truth"):
        return None
    return {
        "question": payload["question"].strip(),
        "ground_truth": payload["ground_truth"].strip(),
        "ood": False,
        "gold_context": chunk_text.strip(),
    }

async def generate_ood_question() -> dict | None:
    prompt = """Generate ONE question that is OUT OF SCOPE for a travel/tourism report.
The question should be answerable in general, but not from the report.
Return a JSON object with keys: question, ground_truth.

Use ground_truth to say the answer is not in the document.

JSON:"""
    completion = await judge_client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": "You create out-of-scope questions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=200,
    )
    content = completion.choices[0].message.content or ""
    match = QA_JSON_PATTERN.search(content)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not payload.get("question"):
        return None
    return {
        "question": payload["question"].strip(),
        "ground_truth": "The document does not contain this information.",
        "ood": True,
    }

async def build_dataset_from_chunks(
    chunks_path: str,
    num_questions: int,
    seed: int,
    max_chars: int,
    ood_ratio: float,
) -> list[dict]:
    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    random.seed(seed)
    ood_ratio = max(0.0, min(ood_ratio, 1.0))
    ood_count = int(round(num_questions * ood_ratio))
    in_scope_count = max(num_questions - ood_count, 0)

    sample = random.sample(chunks, k=min(in_scope_count, len(chunks)))

    sem_gen = asyncio.Semaphore(3)

    async def _generate(chunk: str) -> dict | None:
        async with sem_gen:
            return await generate_qa_from_chunk(chunk[:max_chars])

    tasks = [_generate(chunk) for chunk in sample]
    results = await asyncio.gather(*tasks)
    dataset = [r for r in results if r is not None]

    if ood_count > 0:
        ood_tasks = [generate_ood_question() for _ in range(ood_count)]
        ood_results = await asyncio.gather(*ood_tasks)
        dataset.extend([r for r in ood_results if r is not None])

    random.shuffle(dataset)
    return dataset

# ─────────────────────────────────────────────────────────────────────
# 4. METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def exact_match(pred: str, truth: str) -> int:
    return 1 if normalize_text(pred) == normalize_text(truth) else 0

def token_f1(pred: str, truth: str) -> float:
    pred_tokens = re.findall(r"\w+", pred.lower())
    truth_tokens = re.findall(r"\w+", truth.lower())
    if not pred_tokens or not truth_tokens:
        return 0.0
    pred_counts = {}
    truth_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in truth_tokens:
        truth_counts[t] = truth_counts.get(t, 0) + 1
    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, truth_counts.get(t, 0))
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def retrieval_metrics(gold_context: str | None, retrieved_texts: list[str]) -> dict:
    if not gold_context:
        return {
            "retrieval_hit_rank": None,
            "retrieval_mrr": None,
            "context_precision": None,
            "context_recall": None,
        }
    gold_norm = normalize_text(gold_context)
    hit_rank = None
    for idx, ctx in enumerate(retrieved_texts, start=1):
        ctx_norm = normalize_text(ctx)
        if gold_norm in ctx_norm or ctx_norm in gold_norm:
            hit_rank = idx
            break
    k = max(len(retrieved_texts), 1)
    if hit_rank is None:
        return {
            "retrieval_hit_rank": None,
            "retrieval_mrr": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
    return {
        "retrieval_hit_rank": hit_rank,
        "retrieval_mrr": 1.0 / hit_rank,
        "context_precision": 1.0 / k,
        "context_recall": 1.0,
    }

# ─────────────────────────────────────────────────────────────────────
# 5. FIXED RAG STUDENT (Fixed Tuple Error + Logic for Relevancy)
# ─────────────────────────────────────────────────────────────────────
def get_student_response(question: str) -> tuple[str, list[str], float, list[str]]:
    from vector_store import hybrid_retrieve
    # top_n=5 gives more evidence to increase Faithfulness
    start = time.time()
    docs_with_scores = hybrid_retrieve(question, vs, bm25, chunks_ref, top_n=5)
    retrieval_ms = (time.time() - start) * 1000.0
    
    # FIXED: Correctly unpacking the (Document, Score) tuple
    contexts = [doc.page_content for doc, score in docs_with_scores]
    
    # PROMPT TUNING: Forcing strictness to increase Relevancy/Faithfulness
    prompt = f"""SYSTEM: You are a strict data extractor. Answer ONLY using the provided context.
    If the answer isn't there, say "I cannot find this information in the document."
    Do not mention your training data.

    CONTEXT:
    {" ".join(contexts)}

    QUESTION: {question}
    ANSWER (Concise):"""
    
    answer = llm.invoke(prompt)
    return answer, contexts, retrieval_ms, contexts

# ─────────────────────────────────────────────────────────────────────
# 6. ASYNC EXPERIMENT WITH PARALLEL LIMIT (3 Workers)
# ─────────────────────────────────────────────────────────────────────
# Semaphore ensures we never have more than 3 Ollama calls at once
sem = asyncio.Semaphore(3)

@experiment()
async def run_eval_experiment(row):
    async with sem:
        # Step 1: Run RAG
        answer, contexts, retrieval_ms, retrieved_texts = get_student_response(row["question"])
        gold_context = row.get("gold_context")
        ood = row.get("ood", False)
        rm = retrieval_metrics(None if ood else gold_context, retrieved_texts)
        
        # Step 2: Run Judge
        verdict, reason = await judge_verdict(
            question=row["question"],
            response=answer,
            contexts=contexts,
            ground_truth=row["ground_truth"],
        )

        em = exact_match(answer, row["ground_truth"]) if not ood else None
        f1 = token_f1(answer, row["ground_truth"]) if not ood else None
        refused = bool(REFUSAL_PATTERN.search(answer)) if ood else None
        
        return {
            "question": row["question"],
            "answer": answer,
            "verdict": verdict,
            "reason": reason,
            "score_numeric": 1 if verdict == "Excellent" else 0,
            "retrieval_ms": round(retrieval_ms, 2),
            "ood": ood,
            "gold_context": gold_context,
            "retrieval_hit_rank": rm["retrieval_hit_rank"],
            "retrieval_mrr": rm["retrieval_mrr"],
            "context_precision": rm["context_precision"],
            "context_recall": rm["context_recall"],
            "exact_match": em,
            "f1": f1,
            "refused": refused,
        }

# ─────────────────────────────────────────────────────────────────────
# 7. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="chunks/chunks.jsonl")
    parser.add_argument("--dataset", default="evals/datasets/auto_eval.jsonl")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--ood-ratio", type=float, default=0.3)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dataset), exist_ok=True)

    if args.regenerate or not os.path.exists(args.dataset):
        dataset = await build_dataset_from_chunks(
            chunks_path=args.chunks,
            num_questions=args.num_questions,
            seed=args.seed,
            max_chars=args.max_chars,
            ood_ratio=args.ood_ratio,
        )
        with open(args.dataset, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        dataset = []
        with open(args.dataset, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                dataset.append(json.loads(line))

    print(f"\n📊 Starting Parallel Experiment (3 Workers)...")
    start_time = time.time()

    # Create tasks for parallel execution
    tasks = [run_eval_experiment(row) for row in dataset]
    results = await asyncio.gather(*tasks)

    # Report Generation
    df = pd.DataFrame(results)
    elapsed = round(time.time() - start_time, 2)
    total_n = len(df)
    avg_retrieval_ms = df["retrieval_ms"].mean()
    p95_retrieval_ms = df["retrieval_ms"].quantile(0.95)

    df_in = df[df["ood"] == False]
    df_ood = df[df["ood"] == True]

    in_n = len(df_in)
    ood_n = len(df_ood)

    in_excellent_n = (df_in["verdict"] == "Excellent").sum()
    in_hallucinated_n = (df_in["verdict"] == "Hallucinated").sum()
    in_irrelevant_n = (df_in["verdict"] == "Irrelevant").sum()

    in_relevance_rate = (in_excellent_n / in_n) * 100 if in_n else 0.0
    in_hallucination_rate = (in_hallucinated_n / in_n) * 100 if in_n else 0.0
    in_irrelevance_rate = (in_irrelevant_n / in_n) * 100 if in_n else 0.0
    in_faithfulness_rate = 100.0 - in_hallucination_rate
    in_em = df_in["exact_match"].mean() * 100 if in_n else 0.0
    in_f1 = df_in["f1"].mean() * 100 if in_n else 0.0
    in_recall = df_in["context_recall"].mean() * 100 if in_n else 0.0
    in_mrr = df_in["retrieval_mrr"].mean() if in_n else 0.0
    in_ctx_precision = df_in["context_precision"].mean() * 100 if in_n else 0.0

    ood_refusal_rate = df_ood["refused"].mean() * 100 if ood_n else 0.0
    ood_hallucination_rate = (df_ood["verdict"] == "Hallucinated").sum() / ood_n * 100 if ood_n else 0.0

    print(f"\n{'═'*60}")
    print(
        f"🏁 EVALUATION COMPLETE | Time: {elapsed}s | "
        f"Avg Retrieval: {avg_retrieval_ms:.1f}ms | P95: {p95_retrieval_ms:.1f}ms"
    )
    print(
        f"IN-SCOPE (N={in_n}) | "
        f"Answer Relevance: {in_relevance_rate:.1f}% | "
        f"Faithfulness: {in_faithfulness_rate:.1f}% | "
        f"Hallucination: {in_hallucination_rate:.1f}% | "
        f"Irrelevance: {in_irrelevance_rate:.1f}% | "
        f"EM: {in_em:.1f}% | F1: {in_f1:.1f}% | "
        f"Recall@k: {in_recall:.1f}% | MRR: {in_mrr:.3f} | "
        f"Context Precision: {in_ctx_precision:.1f}%"
    )
    print(
        f"OOD (N={ood_n}) | "
        f"Refusal Rate: {ood_refusal_rate:.1f}% | "
        f"Hallucination: {ood_hallucination_rate:.1f}%"
    )
    print(f"{'═'*60}")
    print(df[["question", "verdict", "answer"]])

    os.makedirs("evals/experiments", exist_ok=True)
    df.to_csv("evals/experiments/fast_eval_report.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())