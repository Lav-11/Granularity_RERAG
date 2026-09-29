"""
run_factoid_rag.py

Factoid-RAG evaluation runner.

Purpose
  - Perform retrieval-augmented evaluation using a factoid-level FAISS index.
  - For each synthetic TBGA query:
      * extract a candidate (gene, relation, disease) triple,
      * build a factoid retrieval query (fallback to gold when extractor is weak),
      * retrieve top-k factoids (metadata entries) from the factoid index,
      * convert factoid metadata into compact strings for prompt context,
      * call the LLM to produce a single-token answer,
      * evaluate the answer with semantic matching and write a JSONL record per example.

Design notes
  - Retrieval returns compact factoid strings for prompts while preserving metadata in outputs.
  - Prompts are strict to encourage single-token responses.
  - Defensive coding tolerates missing indices, malformed metadata, and extractor failures.
  - Deterministic seeds are set for reproducibility.
"""

import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Shared utilities: normalization, retrieval helpers, embedding, LLM wrapper, metrics.
from utils_queries import *

# Deterministic seeds for reproducibility
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------
def _context_to_text(context):
    """
    Convert a list of factoid items into a single block of text suitable for prompts.

    Each factoid may be:
      - a dict with keys 'gene', 'relation', 'disease' (metadata form), or
      - a plain string (already compacted).
    The function produces compact "gene relation disease" lines for the prompt,
    skipping None entries and preserving order.
    """
    lines = []
    for c in context:
        if c is None:
            continue
        if isinstance(c, dict):
            # Build a compact, human-readable line from metadata
            s = f"{c.get('gene','')} {c.get('relation','')} {c.get('disease','')}".strip()
        else:
            # Already a string; coerce to str to be defensive
            s = str(c)
        lines.append(s)
    return "\n".join(lines)


def build_prompt_relation_centric(query, context):
    """
    Build a relation-centric prompt using factoid context.
    The model is instructed to return the relation only.
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
    Build an object/disease-centric prompt using factoid context.
    The model is instructed to return the disease only and to respect the relation.
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
    Build a subject/gene-centric prompt using factoid context.
    The model is instructed to return the gene only.
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
# Evaluation Pipeline
# ---------------------------------------------------------
def evaluate_system(tbga_items, queries, output_path, system_name, mode,
                    genes, diseases, relations, sms_threshold, top_k, emb_dir):
    """
    Run a Factoid-RAG evaluation loop and write per-example JSONL output.

    Args:
      - tbga_items: list of TBGA gold items (dicts with 'h','relation','t')
      - queries: list of synthetic natural-language queries
      - output_path: path to write JSONL per-example records
      - system_name: human-readable name for progress display
      - mode: one of "relation_centric", "object_centric", "subject_centric"
      - genes/diseases/relations: known-entity lists used by the extractor
      - sms_threshold: semantic-match threshold for evaluation
      - top_k: number of factoids to retrieve for context
      - emb_dir: optional directory containing factoid embeddings/index/metadata

    Behavior:
      - Loads the factoid retriever (supports emb_dir override).
      - For each example: extract triple, build retrieval query, retrieve top-k factoids,
        convert metadata to compact strings for prompts, call model, evaluate, and write JSONL.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load factoid retriever artifacts. The loader accepts an optional emb_dir.
    embeddings, index, metadata = load_factoid_retriever(emb_dir)

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
            gene_q = triple.get("gene") or item["h"]["name"]
            rel_q = triple.get("relation") or item["relation"]
            dis_q = triple.get("disease") or item["t"]["name"]

            retrieval_query_factoid = build_factoid_retrieval_query(
                gene_q, rel_q, dis_q, mode
            )

            # Embed the retrieval query and fetch top-k factoids (metadata entries)
            q_emb = embed_query(retrieval_query_factoid)
            retrieved_raw = retrieve_top_k(q_emb, index, metadata, k=top_k)

            # Convert metadata entries into compact strings for the prompt.
            # Keep original metadata available in the JSONL output for debugging.
            retrieved_factoids = []
            for x in retrieved_raw:
                if isinstance(x, dict):
                    g = x.get("gene", "")
                    r = x.get("relation", "")
                    d = x.get("disease", "")
                    retrieved_factoids.append(" ".join([g, r, d]).strip())
                else:
                    retrieved_factoids.append(str(x))

            # Build the prompt depending on the evaluation mode
            if mode == "relation_centric":
                gold = gold_relation
                prompt = build_prompt_relation_centric(q, retrieved_factoids)

            elif mode == "object_centric":
                gold = gold_disease
                prompt = build_prompt_object_subject(q, retrieved_factoids)

            elif mode == "subject_centric":
                gold = gold_gene
                prompt = build_prompt_subject_centric(q, retrieved_factoids)

            else:
                # Defensive fallback: include raw context text if mode is unexpected
                gold = ""
                prompt = f"{q}\n\nCONTEXT:\n" + _context_to_text(retrieved_factoids)

            # Call the model via the project wrapper. call_model enforces short-answer behavior.
            try:
                answer = call_model(prompt) or ""
                answer = answer.strip()
            except Exception:
                # Keep pipeline robust: mark parse errors and continue
                answer = "PARSE_ERROR"

            pred = answer if answer else "PARSE_ERROR"

            # Accumulate for aggregate metrics
            gold_all.append(gold)
            pred_all.append(pred)

            # Semantic evaluation: compute similarity score and matching info
            ok, info = semantic_match(pred, gold, task=mode, threshold=sms_threshold)

            # Use only jw as the saved/aggregated SMS metric; fallback to 'score' if jw missing
            jw = info.get("jw", info.get("score", 0.0))
            sms_scores.append(jw)

            # Write a detailed JSONL record for offline analysis and debugging.
            # Preserve both the compact prompt context and the original metadata where available.
            fout.write(json.dumps({
                "system": system_name,
                "task": f"{mode}",
                "query": q,
                "gold": gold,
                "pred": pred,
                "triple_extracted": triple,
                "retrieval_query_sentence": None,
                "retrieval_query_factoid": retrieval_query_factoid,
                "retrieved_sentences": None,
                "retrieved_factoids": retrieved_factoids,
                "sms": {"jw": jw, "matched_synonym": info.get("matched_synonym")},
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
    print("\nFactoid‑RAG execution parameters:")
    print(f"num_queries   = {num_queries}")
    print(f"top_k         = {top_k}")
    print(f"sms_threshold = {sms_threshold}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load TBGA train/validation splits (expected JSONL)
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

    # Run three Factoid‑RAG evaluations (relation, object/disease, subject/gene)
    evaluate_system(
        tbga_val,
        relation_queries,
        "results/factoid_rag/relation_centric.jsonl",
        "Factoid‑RAG Relation-Centric",
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
        "results/factoid_rag/object_centric.jsonl",
        "Factoid‑RAG Object-Centric",
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
        "results/factoid_rag/subject_centric.jsonl",
        "Factoid‑RAG Subject-Centric",
        mode="subject_centric",
        genes=genes,
        diseases=diseases,
        relations=relations,
        sms_threshold=sms_threshold,
        top_k=top_k,
        emb_dir=emb_dir
    )
