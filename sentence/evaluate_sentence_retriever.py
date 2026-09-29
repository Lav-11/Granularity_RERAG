import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------
# SCRIPT_DIR: directory where this script lives
# EMB_DIR: folder containing the FAISS index and metadata produced from TRAIN sentence embeddings
# EVAL_DIR: folder containing evaluation files (VAL / TEST)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_DIR = os.path.join(SCRIPT_DIR, "data", "embeddings_sentence")
EVAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "eval"))

# Files expected to exist
FAISS_INDEX = os.path.join(EMB_DIR, "index.faiss")
META_JSONL = os.path.join(EMB_DIR, "metadata.jsonl")

# Evaluation files (VAL / TEST) expected to contain triples for evaluation
VAL_EVAL_FILE = os.path.join(EVAL_DIR, "tbga_val_eval.jsonl")
TEST_EVAL_FILE = os.path.join(EVAL_DIR, "tbga_test_eval.jsonl")

# ---------------------------------------------------------
# Model for encoding queries
# ---------------------------------------------------------
# This model is used to encode the query factoid (gene relation disease) into an embedding.
# We use a lightweight SentenceTransformer for query encoding; it must be compatible with
# the embedding space used to build the FAISS index (or at least produce comparable vectors).
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------
# Helpers to load metadata and evaluation data
# ---------------------------------------------------------
def load_metadata(path):
    """
    Load TRAIN sentence metadata saved as JSONL.
    Each line is expected to be a JSON object with keys like 'sentence','gene','disease','relation'.
    Returns a list of metadata dicts in the same order as the FAISS index.
    """
    metadata = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def load_eval_data(path):
    """
    Load evaluation triples (VAL or TEST) saved as JSONL.
    Each line is expected to be a JSON object with keys 'gene','relation','disease'.
    Returns a list of evaluation items.
    """
    eval_items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            eval_items.append(json.loads(line))
    return eval_items

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def compute_metrics(ranks):
    """
    Compute Recall@1, Recall@5, Recall@10 and MRR from a list/array of ranks.
    Ranks should be 1-based (1 means top-1), and a large sentinel (e.g., 9999) indicates not found.
    """
    ranks = np.array(ranks)
    return (
        np.mean(ranks <= 1),
        np.mean(ranks <= 5),
        np.mean(ranks <= 10),
        np.mean(1.0 / ranks)
    )

# ---------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading FAISS index...")
    # Load the FAISS index built over TRAIN sentence embeddings
    index = faiss.read_index(FAISS_INDEX)

    print("Loading metadata...")
    # Load TRAIN metadata (one entry per indexed vector)
    metadata = load_metadata(META_JSONL)

    print("Loading evaluation data (VAL)...")
    # Load VAL triples (gold queries)
    eval_items = load_eval_data(VAL_EVAL_FILE)

    print(f"Loaded {len(eval_items)} evaluation queries")

    # Precompute TRAIN triples as compact strings for exact matching against retrieved items.
    # IMPORTANT: these triples are derived from TRAIN metadata only.
    index_triples = [
        f"{m['gene'].upper()} {m['relation']} {m['disease'].lower().strip()}"
        for m in metadata
    ]

    ranks = []

    # Iterate over each VAL triple and perform retrieval
    for idx_item, item in enumerate(eval_items):

        # GOLD triple (from VAL)
        # Note: VAL triples typically do not appear verbatim in TRAIN, so exact-match evaluation
        # against TRAIN triples will usually fail unless the dataset intentionally overlaps.
        gold = f"{item['gene'].upper()} {item['relation']} {item['disease'].lower().strip()}"

        # Build the retrieval query (factoid string)
        query = f"{item['gene']} {item['relation']} {item['disease']}"

        # Encode the query into an embedding using the SentenceTransformer model
        # normalize_embeddings=True ensures vectors are L2-normalized (useful for cosine similarity)
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Retrieve top-10 nearest TRAIN sentence vectors from the FAISS index
        D, I = index.search(q_emb, 10)

        # Map retrieved indices to TRAIN triples
        retrieved_triples = [
            index_triples[idx] for idx in I[0]
        ]

        # EXACT MATCH CHECK
        # This checks whether the gold VAL triple exactly matches any retrieved TRAIN triple.
        # Because VAL triples are usually absent from TRAIN, this will often be false.
        if gold in retrieved_triples:
            rank = retrieved_triples.index(gold) + 1
        else:
            rank = 9999  # sentinel for "not found"

        ranks.append(rank)

        # Print a few debug examples to inspect behavior
        if idx_item < 3:
            print("\n=== EXAMPLE", idx_item, "===")
            print("QUERY:", query)
            print("GOLD TRIPLE:", gold)
            print("TOP-10 TRIPLES:", retrieved_triples)
            print("RANK:", rank)

    # Compute aggregate metrics from ranks
    recall1, recall5, recall10, mrr = compute_metrics(ranks)

    print("\n=== RETRIEVAL RESULTS (FACTOID → SENTENCE, EXACT MATCH) ===")
    print(f"Recall@1:  {recall1:.4f}")
    print(f"Recall@5:  {recall5:.4f}")
    print(f"Recall@10: {recall10:.4f}")
    print(f"MRR:       {mrr:.4f}")
