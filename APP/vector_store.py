import os
import json
import pickle
import numpy as np
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────
def load_passed_chunks(path="chunks/chunks_processed.jsonl"):
    # Robust path checking: check current and parent dir
    potential_paths = [Path(path), Path("../") / path]
    target_path = None
    for p in potential_paths:
        if p.exists():
            target_path = p
            break
            
    if not target_path:
        print(f"❌ Error: Could not find {path}")
        return []

    passed_chunks = []
    print(f"🔄 Loading data from: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Only index what passed our Phase 3 gate
            if data["metadata"].get("passed_gate") is True:
                passed_chunks.append(Document(
                    page_content=data["page_content"],
                    metadata=data["metadata"]
                ))
    return passed_chunks

# ─────────────────────────────────────────────────────────────────────
# 2. BUILD INDICES
# ─────────────────────────────────────────────────────────────────────
def build_hybrid_indices(chunks):
    if not chunks:
        print("⚠️ No chunks to index. Skipping build.")
        return None, None

    # Create directory for indexes
    idx_dir = Path("indexes")
    idx_dir.mkdir(exist_ok=True)

    # A. Build FAISS (Semantic)
    print("🧠 Building FAISS Semantic Index (Dense)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(idx_dir / "faiss_index"))

    # B. Build BM25 (Keyword)
    print("📝 Building BM25 Keyword Index (Sparse)...")
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(idx_dir / "bm25_data.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        
    print(f"✅ All indices built. Saved to: {idx_dir.absolute()}")
    return vectorstore, bm25

# ─────────────────────────────────────────────────────────────────────
# 3. HYBRID RETRIEVAL (RRF)
# ─────────────────────────────────────────────────────────────────────
def hybrid_retrieve(query, vectorstore, bm25, chunks, k=60, top_n=3):
    # 1. Semantic Search
    semantic_results = vectorstore.similarity_search(query, k=10)
    
    # 2. Keyword Search
    tokenized_query = query.lower().split()
    keyword_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(keyword_scores)[::-1][:10]
    keyword_results = [chunks[i] for i in top_indices if keyword_scores[i] > 0]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(semantic_results, 1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)

    for rank, doc in enumerate(keyword_results, 1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[cid], score) for cid, score in sorted_results[:top_n]]

# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 STARTING PHASE 5: VECTOR STORE & HYBRID SEARCH")
    print("="*50)
    
    # 1. Load Data
    all_passed = load_passed_chunks()
    print(f"📥 Found {len(all_passed)} high-quality chunks.")
    
    if all_passed:
        # 2. Build
        vs, bm = build_hybrid_indices(all_passed)

        # 3. Test Retrieval
        test_query = input("\nEnter a test query to retrieve relevant chunks: ")
        print(f"\n🔍 Testing Hybrid Search: '{test_query}'")
        
        results = hybrid_retrieve(test_query, vs, bm, all_passed)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[{i}] RRF Score: {score:.4f} | Page: {doc.metadata.get('page')}")
            print(f"Content: {doc.page_content[:150]}...")
    else:
        print("❌ Build stopped: No data loaded. Check chunks/chunks_processed.jsonl")