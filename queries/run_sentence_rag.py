"""
run_sentence_rag.py

Sentence‑RAG evaluation runner.

This script performs retrieval-augmented evaluation using a sentence-level FAISS index.
For each synthetic TBGA query it:
  - extracts a candidate (gene, relation, disease) triple using the local extractor,
  - builds a retrieval query (falling back to gold values when necessary),
  - retrieves top-k sentences from the sentence index,
  - constructs a concise prompt containing the query and retrieved context,
  - calls the model to produce a single-token answer (relation / disease / gene),
  - evaluates the answer with semantic matching and writes a JSONL record per example.

Design notes:
  - The runner is defensive: it tolerates missing indices and malformed metadata.
  - Retrieval returns only sentence text for prompts; metadata is preserved in the retriever.
  - Prompts are intentionally strict to encourage parsable single-token outputs.
  - The script is intended to be run from the `queries` directory and to interoperate
    with the rest of the pipeline (launcher, utils_queries).
"""

import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Import shared utilities: normalization, retrieval helpers, embedding, LLM wrapper, metrics.
from utils_queries import *

# Deterministic seeds for reproducibility
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# Prompt builders (sentence-level RAG)
# ---------------------------------------------------------
# Each builder returns a compact prompt that includes the original query and
# a small context (top-k retrieved sentences). Prompts instruct the model to
# return a single short token only (no explanations, synonyms, or extra text).
def _context_to_text(context):
    """
    Convert a list of context strings into a single block of text.
    The function assumes `context` is a list of plain strings (sentences).
    """
    return "\n".join(context)


def build_prompt_relation_centric(query, context):
    """
    Build a relation-centric prompt: model must return the relation only.
    """
    ctx = _context_to_text(context)
    return (
        f"{query}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        "INSTRUCTIONS:\n"
        "- Respond with the relation only.\n"
        "- No explanations.\n"
        "- No synonyms.\n"
        "- No additional text.\n"
    )


def build_prompt_object_subject(query, context):
    """
    Build an object/disease-centric prompt: model must return the disease only.
    """
    ctx = _context_to_text(context)
    return (
        f"{query}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        "INSTRUCTIONS:\n"
        "- Respond with the disease only.\n"
        "- Use the relation exactly as written.\n"
        "- No explanations.\n"
        "- No synonyms.\n"
        "- No additional text.\n"
    )


def build_prompt_subject_centric(query, context):
    """
    Build a subject/gene-centric prompt: model must return the gene only.
    """
    ctx = _context_to_text(context)
    return (
        f"{query}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        "INSTRUCTIONS:\n"
        "- Respond with the gene only.\n"
        "- Use the relation exactly as written.\n"
        "- No explanations.\n"
        "- No synonyms.\n"
        "- No additional text.\n"
    )


# ---------------------------------------------------------
# Evaluation pipeline (Sentence‑RAG)
# ---------------------------------------------------------
def evaluate_system(tbga_items, queries, output_path, system_name, mode,
                    genes, diseases, relations, sms_threshold, top_k, emb_dir):
    """
    Run a Sentence‑RAG evaluation loop and write per-example JSONL output.

    Args:
      tbga_items: list of TBGA gold items (dicts with 'h','relation','t')
      queries: list of synthetic natural-language queries
      output_path: path to write JSONL per-example records
      system_name: human-readable name for progress display
      mode: one of "relation_centric", "object_centric", "subject_centric"
      genes/diseases/relations: known-entity lists used by the extractor
      sms_threshold: semantic-match threshold for evaluation
      top_k: number of sentences to retrieve for context
      emb_dir: optional directory containing sentence embeddings/index/metadata

    Behavior:
      - Loads the sentence retriever (supports emb_dir override).
      - For each example: extract triple, build retrieval query, retrieve top-k sentences,
        build prompt, call model, evaluate, and write a JSONL record.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load sentence retriever artifacts. The loader accepts an optional emb_dir.
    embeddings, index, metadata = load_sentence_retriever(emb_dir)

    gold_all, pred_all, sms_scores = [], [], []

    # Stream JSONL output for each example to facilitate offline analysis
    with open(output_path, "w", encoding="utf-8") as fout:

        # Iterate in lockstep over items and queries; tqdm provides a progress bar.
        for item, q in tqdm(zip(tbga_items, queries), total=len(tbga_items), desc=system_name):

            # Normalize gold labels from TBGA for consistent comparison
            gold_gene = normalize_gene(item["h"]["name"])
            gold_relation = normalize_with_synonyms(item["relation"], RELATION_SYNONYMS)
            gold_disease = item["t"]["name"].lower().strip()

            # Extract a candidate triple from the question using the local extractor.
            # The extractor is dictionary-first and may call an LLM only for missing fields.
            triple = extract_triplet_from_question(q, genes, diseases, relations, mode)

            # Build retrieval query from the extracted triple; fall back to gold values
            # when the extractor did not find a reliable field. This keeps retrieval robust.
            gene_q = triple.get("gene") or item["h"]["name"]
            rel_q = triple.get("relation") or item["relation"]
            dis_q = triple.get("disease") or item["t"]["name"]

            retrieval_query_sentence = build_sentence_retrieval_query(
                gene_q, rel_q, dis_q, mode
            )

            # Embed the retrieval query and fetch top-k sentences
            query_vec = embed_query(retrieval_query_sentence)

            # retrieve_top_k returns metadata entries; convert to plain sentence strings
            # for prompt construction. This preserves the original JSONL output format
            # while ensuring prompts contain only human-readable sentences.
            raw_hits = retrieve_top_k(query_vec, index, metadata, k=top_k)
            retrieved_sentences = [hit.get("sentence", "") for hit in raw_hits]

            # Build the prompt depending on the evaluation mode
            if mode == "relation_centric":
                gold = gold_relation
                prompt = build_prompt_relation_centric(q, retrieved_sentences)

            elif mode == "object_centric":
                gold = gold_disease
                prompt = build_prompt_object_subject(q, retrieved_sentences)

            elif mode == "subject_centric":
                gold = gold_gene
                prompt = build_prompt_subject_centric(q, retrieved_sentences)

            else:
                # Defensive fallback: include raw context text if mode is unexpected
                gold = ""
                prompt = f"{q}\n\nCONTEXT:\n" + _context_to_text(retrieved_sentences)

            # Call the model via the project wrapper. call_model enforces short-answer behavior.
            answer = call_model(prompt).strip()
            pred = answer if answer else "PARSE_ERROR"

            # Accumulate for aggregate metrics
            gold_all.append(gold)
            pred_all.append(pred)

            # Semantic evaluation: compute similarity score and matching info
            ok, info = semantic_match(pred, gold, task=mode, threshold=sms_threshold)
            sms_scores.append(info["score"])

            # Write a detailed JSONL record for offline analysis and debugging.
            # Keep the same record structure used elsewhere in the project.
            fout.write(json.dumps({
                "system": "sentence_rag",
                "task": f"{mode}",
                "query": q,
                "gold": gold,
                "pred": pred,
                "triple_extracted": triple,
                "retrieval_query_sentence": retrieval_query_sentence,
                "retrieval_query_factoid": None,
                "retrieved_sentences": retrieved_sentences,   # list[str]
                "retrieved_factoids": None,
                "sms": info,
                "matched_synonym": info.get("matched_synonym")
            }, ensure_ascii=False) + "\n")

    # Compute aggregate metrics (Exact Match, Semantic Accuracy, Precision, Recall)
    metrics = compute_metrics(gold_all, pred_all, sms_scores, sms_threshold=sms_threshold)

    # Print a concise summary for quick inspection
    print(f"\n=== {system_name} — METRICS ===")
    print("Exact Match (EM):", metrics["em"])
    print("Semantic Accuracy (SMS ≥ threshold):", metrics["sms"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print()

    return metrics


# ---------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Execution controls
    parser.add_argument("--num_queries", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sms_threshold", type=float, default=0.80)
    parser.add_argument("--force_rebuild_indices", action="store_true")
    parser.add_argument("--hgnc_path", type=str, default=None)
    parser.add_argument("--emb_dir", type=str, default=None)

    args = parser.parse_args()

    num_queries = args.num_queries
    top_k = args.top_k
    sms_threshold = args.sms_threshold
    force_rebuild = args.force_rebuild_indices
    hgnc_path = args.hgnc_path
    emb_dir = args.emb_dir

    # Execution summary printed for traceability
    print("\nSentence‑RAG execution parameters:")
    print(f"num_queries   = {num_queries}")
    print(f"top_k         = {top_k}")
    print(f"sms_threshold = {sms_threshold}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load TBGA train/validation splits (expected to be JSONL)
    with open(os.path.normpath(os.path.join(BASE_DIR, "..", "..", "benchmark", "TBGA", "TBGA_train.txt")), encoding="utf-8") as f:
        tbga_train = [json.loads(line) for line in f]

    with open(os.path.normpath(os.path.join(BASE_DIR, "..", "..", "benchmark", "TBGA", "TBGA_val.txt")), encoding="utf-8") as f:
        tbga_val_full = [json.loads(line) for line in f]

    # Filter valid examples and limit to requested number
    tbga_val = [x for x in tbga_val_full if x.get("relation", "NA") != "NA"][:num_queries]

    # Ensure indices exist if the helper is available; ignore failures in this runner
    try:
        ensure_indices(force_rebuild=force_rebuild)
    except Exception:
        pass

    # Load MeSH descriptors and supplementary records into a combined dictionary
    mesh_desc = load_mesh_dict(os.path.normpath(os.path.join(BASE_DIR, "..", "mesh", "desc2026.xml")))
    mesh_supp = load_mesh_dict(os.path.normpath(os.path.join(BASE_DIR, "..", "mesh", "supp2026.xml")))
    mesh_dict = {**mesh_desc, **mesh_supp}

    # Load HGNC gene file (allow explicit override via --hgnc_path)
    if hgnc_path:
        hgnc_file = os.path.normpath(hgnc_path)
    else:
        hgnc_file = os.path.normpath(os.path.join(BASE_DIR, "..", "genes", "hgnc_complete_set.txt"))

    gene_dict = load_gene_dict(hgnc_file)

    # Build known-entity lists used by the extractor and other helpers
    genes, diseases, relations = load_known_entities(mesh_dict, gene_dict)

    # Generate synthetic queries for the benchmark tasks
    relation_queries = generate_relation_queries_for_items(tbga_val)
    object_queries   = generate_object_queries_for_items(tbga_val)
    subject_queries  = generate_subject_queries_for_items(tbga_val)

    # Run three Sentence‑RAG evaluations (relation, object/disease, subject/gene)
    evaluate_system(
        tbga_val,
        relation_queries,
        "results/sentence_rag/relation_centric.jsonl",
        "Sentence‑RAG Relation-Centric",
        mode="relation_centric",
        genes=genes,
        diseases=diseases,
        relations=relations,
        sms_threshold=sms_threshold,
        top_k=top_k,
        emb_dir=emb_dir
    )

    evaluate_system(
        tbga_val,
        object_queries,
        "results/sentence_rag/object_centric.jsonl",
        "Sentence‑RAG Object-Centric",
        mode="object_centric",
        genes=genes,
        diseases=diseases,
        relations=relations,
        sms_threshold=sms_threshold,
        top_k=top_k,
        emb_dir=emb_dir
    )

    evaluate_system(
        tbga_val,
        subject_queries,
        "results/sentence_rag/subject_centric.jsonl",
        "Sentence‑RAG Subject-Centric",
        mode="subject_centric",
        genes=genes,
        diseases=diseases,
        relations=relations,
        sms_threshold=sms_threshold,
        top_k=top_k,
        emb_dir=emb_dir
    )
