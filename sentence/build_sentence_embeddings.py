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
# DATA_DIR: top-level data folder (contains jsonl_sentences and embeddings_sentence)
# SENTENCE_DIR: input JSONL with one sentence per line (expected keys: sentence, gene, disease, relation)
# EMB_DIR: output folder for embeddings, metadata and faiss index
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SENTENCE_DIR = os.path.join(DATA_DIR, "jsonl_sentences")
EMB_DIR = os.path.join(DATA_DIR, "embeddings_sentence")

os.makedirs(EMB_DIR, exist_ok=True)

# Input JSONL expected format: each line is a JSON object with at least "sentence","gene","disease","relation"
INPUT_FILE = os.path.join(SENTENCE_DIR, "tbga_train_sentences.jsonl")

# Output artifacts
EMB_NPY = os.path.join(EMB_DIR, "sentence_embeddings.npy")
META_JSONL = os.path.join(EMB_DIR, "metadata.jsonl")
FAISS_INDEX = os.path.join(EMB_DIR, "index.faiss")


# ---------------------------------------------------------
# Contriever model loading
# ---------------------------------------------------------
# We use facebook/contriever (sentence-level encoder). Tokenizer + model are loaded once.
# Model is set to eval() mode. Device placement happens at encoding time.
MODEL_NAME = "facebook/contriever"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


# ---------------------------------------------------------
# Sentence encoding
# ---------------------------------------------------------
def contriever_encode(texts, batch_size=64):
    """
    Encode a list of sentences using Contriever.

    - Uses GPU if available (torch.cuda.is_available()).
    - Returns a numpy array of shape (N, D) where D is the model hidden size (768 for Contriever).
    - Uses the [CLS] token embedding (outputs.last_hidden_state[:, 0, :]) as a sentence vector.
    - Batch processing with tokenizer padding/truncation for efficiency.

    Args:
      texts (List[str]): list of sentences to encode
      batch_size (int): batch size for tokenization / model forward pass

    Returns:
      np.ndarray: stacked embeddings (float32)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embeddings = []

    # No grad for inference
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

            # Use CLS token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

    # Stack all batches into a single matrix
    return np.vstack(all_embeddings)


# ---------------------------------------------------------
# Load sentences and metadata
# ---------------------------------------------------------
def load_sentences(path):
    """
    Read a JSONL file and extract sentences and metadata.

    Expected JSONL schema per line:
      {
        "sentence": "text ...",
        "gene": "BRCA1",
        "disease": "ovarian cancer",
        "relation": "associated_with"
      }

    Returns:
      sentences: list[str] (raw sentence text)
      metadata: list[dict] (keeps sentence + gene/disease/relation for later retrieval)
    """
    sentences = []
    metadata = []

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            data = json.loads(line)

            sentence = data["sentence"]
            sentences.append(sentence)

            # Keep a compact metadata dict per sentence for later use in retrieval
            metadata.append({
                "sentence": sentence,
                "gene": data.get("gene", ""),
                "disease": data.get("disease", ""),
                "relation": data.get("relation", "")
            })

    return sentences, metadata


# ---------------------------------------------------------
# FAISS index construction
# ---------------------------------------------------------
def build_faiss_index(embeddings):
    """
    Build a FAISS index using inner-product similarity.

    - IndexFlatIP is used for exact inner-product search.
    - If you want cosine similarity, ensure embeddings are L2-normalized before adding.
    - Returns a FAISS index object.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


# ---------------------------------------------------------
# Main script: load, encode, save artifacts
# ---------------------------------------------------------
if __name__ == "__main__":

    print(f"Loading sentences from: {INPUT_FILE}")
    sentences, metadata = load_sentences(INPUT_FILE)
    print(f"Loaded {len(sentences)} sentences")

    print("Encoding sentences using Contriever...")
    emb = contriever_encode(sentences)
    print(f"Embedding matrix shape: {emb.shape}")

    # Save embeddings as numpy .npy for fast reload
    print(f"Saving embeddings to: {EMB_NPY}")
    np.save(EMB_NPY, emb.astype(np.float32))

    # Save metadata as JSONL (one metadata dict per line)
    print(f"Saving metadata to: {META_JSONL}")
    with open(META_JSONL, "w", encoding="utf-8") as fout:
        for m in metadata:
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Build FAISS index and persist it
    print("Building FAISS index...")
    index = build_faiss_index(emb.astype(np.float32))

    print(f"Saving FAISS index to: {FAISS_INDEX}")
    faiss.write_index(index, FAISS_INDEX)

    print("\nSentence embedding index built successfully.")
