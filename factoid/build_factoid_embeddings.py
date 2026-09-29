import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

# ---------------------------------------------------------
# Base paths and I/O configuration
# ---------------------------------------------------------
# SCRIPT_DIR: directory where this script lives
# DATA_DIR: top-level data folder (contains jsonl_factoid and embeddings_factoid)
# FACTOID_DIR: input JSONL with one factoid per line (expected keys: factoid, gene, relation, disease, sentence)
# EMB_DIR: output folder for embeddings, metadata and faiss index
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FACTOID_DIR = os.path.join(DATA_DIR, "jsonl_factoid")
EMB_DIR = os.path.join(DATA_DIR, "embeddings_factoid")

os.makedirs(EMB_DIR, exist_ok=True)

# Input JSONL expected format: each line is a JSON object with at least "factoid","gene","relation","disease","sentence"
INPUT_FILE = os.path.join(FACTOID_DIR, "tbga_train_factoids.jsonl")

# Output artifacts
EMB_NPY = os.path.join(EMB_DIR, "embeddings.npy")
META_JSONL = os.path.join(EMB_DIR, "metadata.jsonl")
FAISS_INDEX = os.path.join(EMB_DIR, "index.faiss")


# ---------------------------------------------------------
# Contriever model loading
# ---------------------------------------------------------
# We use facebook/contriever (sentence/factoid encoder). Tokenizer + model are loaded once.
# Model is set to eval() mode. Device placement happens at encoding time.
MODEL_NAME = "facebook/contriever"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


# ---------------------------------------------------------
# Factoid encoding
# ---------------------------------------------------------
def contriever_encode(texts, batch_size=64):
    """
    Encode a list of factoid strings using Contriever.

    - Uses GPU if available (torch.cuda.is_available()).
    - Returns a numpy array of shape (N, D) where D is the model hidden size.
    - Uses the [CLS] token embedding (outputs.last_hidden_state[:, 0, :]) as a vector.
    - Batch processing with tokenizer padding/truncation for efficiency.

    Args:
      texts (List[str]): list of factoid strings to encode
      batch_size (int): batch size for tokenization / model forward pass

    Returns:
      np.ndarray: stacked embeddings (float32)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch = texts[i:i + batch_size]

            # Tokenize batch; return_tensors="pt" yields PyTorch tensors
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)

            # Use CLS token embedding as representation
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

    # Stack all batches into a single matrix
    return np.vstack(all_embeddings)


# ---------------------------------------------------------
# Load factoids and metadata
# ---------------------------------------------------------
def load_factoids(path):
    """
    Load factoid strings and associated metadata from a JSONL file.

    Expected JSONL schema per line:
      {
        "factoid": "BRCA1 associated_with ovarian cancer",
        "gene": "BRCA1",
        "relation": "associated_with",
        "disease": "ovarian cancer",
        "sentence": "Original sentence text..."
      }

    Returns:
      texts: list[str] (factoid strings)
      meta: list[dict] (keeps factoid + gene/relation/disease/sentence for later retrieval)
    """
    print("Loading factoids...")
    texts = []
    meta = []

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            data = json.loads(line)

            factoid = data["factoid"]
            texts.append(factoid)

            meta.append({
                "factoid": factoid,
                "gene": data.get("gene", ""),
                "relation": data.get("relation", ""),
                "disease": data.get("disease", ""),
                "sentence": data.get("sentence", "")
            })

    return texts, meta


# ---------------------------------------------------------
# FAISS index construction
# ---------------------------------------------------------
def build_index(embeddings):
    """
    Build a FAISS index using inner-product similarity.

    - IndexFlatIP is used for exact inner-product search.
    - If you want cosine similarity, ensure embeddings are L2-normalized before adding.
    - Returns a FAISS index object.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ---------------------------------------------------------
# Main script: load, encode, save artifacts
# ---------------------------------------------------------
if __name__ == "__main__":

    # Load factoid strings and metadata
    texts, meta = load_factoids(INPUT_FILE)

    print("Encoding factoids using Contriever...")
    emb = contriever_encode(texts)
    print(f"Embedding matrix shape: {emb.shape}")

    # Save embeddings as numpy .npy for fast reload
    print("Saving embeddings...")
    np.save(EMB_NPY, emb.astype(np.float32))

    # Save metadata as JSONL (one metadata dict per line)
    print("Saving metadata...")
    with open(META_JSONL, "w", encoding="utf-8") as f:
        for m in tqdm(meta, desc="Writing metadata"):
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Build FAISS index and persist it
    print("Building FAISS index...")
    index = build_index(emb.astype(np.float32))
    faiss.write_index(index, FAISS_INDEX)

    print("\nFactoid embedding index built successfully.")
