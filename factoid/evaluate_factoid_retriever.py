import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------
# SCRIPT_DIR: directory where this script lives
# EMB_DIR: folder containing the FAISS index and metadata produced from TRAIN factoid embeddings
# EVAL_DIR: folder containing evaluation files (VAL / TEST)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# embeddings MUST be in: factoid/data/embeddings/
EMB_DIR = os.path.join(SCRIPT_DIR, "data", "embeddings")

# eval files in eval/
EVAL_DIR = os.path.join(SCRIPT_DIR, "..", "eval")

# Files expected to exist
FAISS_INDEX = os.path.join(EMB_DIR, "index.faiss")
META_JSONL = os.path.join(EMB_DIR, "metadata.jsonl")

# Evaluation files (VAL / TEST) expected to contain factoid triples for evaluation
VAL_EVAL_FILE = os.path.join(EVAL_DIR, "tbga_val_eval.jsonl")
TEST_EVAL_FILE = os.path.join(EVAL_DIR, "tbga_test_eval.jsonl")

# ---------------------------------------------------------
# Query encoder
# ---------------------------------------------------------
# This model encodes the query factoid (e.g., "BRCA1 associated_with ovarian cancer")
# into a vector compatible with the FAISS index. Ensure the encoder is compatible
# with the model used to build the index (or that embeddings are comparable).
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------
# Helpers to load metadata and evaluation data
# ---------------------------------------------------------
def load_metadata(path):
    """
    Load metadata saved as JSONL for the indexed factoids.
    Each line is expected to be a JSON object with keys like:
      'factoid', 'gene', 'relation', 'disease', 'sentence'
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

    recall1 = np.mean(ranks <= 1)
    recall5 = np.mean(ranks <= 5)
    recall10 = np.mean(ranks <= 10)
    mrr = np.mean(1.0 / ranks)

    return recall1, recall5, recall10, mrr


# ---------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading FAISS index...")
    # Load the FAISS index built over TRAIN factoid embeddings
    index = faiss.read_index(FAISS_INDEX)

    print("Loading metadata...")
    # Load indexed factoid metadata (one entry per indexed vector)
    metadata = load_metadata(META_JSONL)

    print("Loading evaluation data...")
    # Load VAL triples (gold queries)
    eval_items = load_eval_data(VAL_EVAL_FILE)

    print(f"Loaded {len(eval_items)} evaluation queries")

    # Precompute factoid strings in the index for quick membership checks
    index_factoids = [m.get("factoid", "") for m in metadata]
    print(f"Factoids in index: {len(index_factoids)}")

    ranks = []
    skipped_na = 0
    skipped_missing = 0

    for idx_item, item in enumerate(eval_items):
        gene = item.get("gene", "")
        disease = item.get("disease", "")
        relation = item.get("relation", "")

        # Skip queries labeled as NA (no relation)
        if relation == "NA":
            skipped_na += 1
            continue

        # Build the canonical factoid string used in index and queries
        gold_factoid = f"{gene} {relation} {disease}"

        # If the gold factoid is not present in the index, skip the example
        # (this script evaluates only examples whose gold exists in the index)
        if gold_factoid not in index_factoids:
            skipped_missing += 1
            continue

        # Encode the query factoid into an embedding
        query = gold_factoid  # same format used for index entries
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Search top-10 nearest neighbors in the FAISS index
        D, I = index.search(q_emb, 10)

        # Map retrieved indices to factoid strings
        retrieved = [metadata[idx]["factoid"] for idx in I[0]]

        # Compute rank of the gold factoid among retrieved items (1-based)
        if gold_factoid in retrieved:
            rank = retrieved.index(gold_factoid) + 1
        else:
            rank = 9999  # sentinel for "not found"

        ranks.append(rank)

        # Print a few debug examples to inspect behavior
        if len(ranks) <= 3:
            print("\n=== EXAMPLE ===")
            print("QUERY:", query)
            print("GOLD:", gold_factoid)
            print("TOP-10:", retrieved)
            print("RANK:", rank)

    # Report how many queries were skipped and compute metrics if any ranks exist
    print(f"\nQuery NA skipped: {skipped_na}")
    print(f"Query with missing gold factoid skipped: {skipped_missing}")

    if len(ranks) == 0:
        print("\nNo valid queries to evaluate.")
    else:
        recall1, recall5, recall10, mrr = compute_metrics(ranks)

        print("\n=== RETRIEVAL RESULTS ===")
        print(f"Recall@1:  {recall1:.4f}")
        print(f"Recall@5:  {recall5:.4f}")
        print(f"Recall@10: {recall10:.4f}")
        print(f"MRR:       {mrr:.4f}")


