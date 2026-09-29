# launcher_quick.py
# Lightweight orchestrator for No‑Context evaluation experiments.
# This script runs the no-context evaluation over a TBGA validation split,
# calling the local runner that queries the LLM without retrieval augmentation.
# It is intended for fast iterations and debugging, not full pipeline orchestration.

import warnings
warnings.filterwarnings("ignore")

import os
import json
import random
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

# Import utilities from utils_queries. The module exposes normalization,
# extraction, metric and helper functions used by the evaluation loop.
from utils_queries import *  # noqa: F401,F403

# Deterministic seeds for reproducibility
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# Prompt builders for No‑Context experiments
# ---------------------------------------------------------
# Each builder returns a compact prompt that instructs the model to output
# a single, parsable token. Prompts are intentionally strict to simplify
# downstream parsing and automatic scoring.
def build_prompt_relation_centric(query):
    return (
        f"Query: {query}\n\n"
        "IMPORTANT RULES:\n"
        "- Answer with ONLY the relation.\n"
        "- No explanations.\n"
        "- No synonyms.\n"
        "- No extra words.\n"
    )

def build_prompt_object_subject(query):
    return (
        f"Query: {query}\n\n"
        "IMPORTANT RULES:\n"
        "- Answer with ONLY the disease.\n"
        "- Use the relation EXACTLY as written.\n"
        "- No explanations.\n"
        "- No synonyms.\n"
        "- No extra words.\n"
    )

def build_prompt_subject_centric(query):
    return (
        f"Query: {query}\n\n"
        "IMPORTANT RULES:\n"
        "- Answer with ONLY the gene.\n"
        "- Use the relation EXACTLY as written.\n"
        "- No explanations.\n"
        "- No synonyms.\n"
        "- No extra words.\n"
    )


# ---------------------------------------------------------
# Evaluation loop (No‑Context)
# ---------------------------------------------------------
# This function executes the no-context evaluation for a single task mode.
# It:
#  - iterates over TBGA items and synthetic queries
#  - extracts a candidate triple using the local extractor
#  - builds a strict prompt and calls the model
#  - computes semantic matching and writes per-example JSONL records
def evaluate_system(tbga_items, queries, output_path, system_name, mode,
                    sms_threshold, genes, diseases, relations):

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gold_all, pred_all, sms_scores = [], [], []

    # Process at most the number of available queries or TBGA items
    n = min(len(tbga_items), len(queries))

    # Stream JSONL output to file for offline analysis
    with open(output_path, "w", encoding="utf-8") as fout:

        for i in tqdm(range(n), desc=system_name):
            item = tbga_items[i]
            q = queries[i]

            # Normalize gold labels for consistent comparison
            gold_gene = normalize_gene(item["h"]["name"])
            gold_relation = normalize_with_synonyms(item["relation"], RELATION_SYNONYMS)
            gold_disease = item["t"]["name"].lower().strip()

            # Triple extraction is dictionary-first and may call LLM only for missing fields
            try:
                triple = extract_triplet_from_question(q, genes, diseases, relations, mode)
            except Exception:
                # Keep the pipeline robust: log a placeholder triple and continue
                triple = {
                    "gene": None,
                    "relation": None,
                    "disease": None,
                    "llm_used": False,
                    "llm_raw_output": None,
                    "source": {"gene": "error", "relation": "error", "disease": "error"},
                }

            # Select prompt and gold target depending on the evaluation mode
            if mode == "relation_centric":
                gold = gold_relation
                prompt = build_prompt_relation_centric(q)

            elif mode == "object_centric":
                gold = gold_disease
                prompt = build_prompt_object_subject(q)

            elif mode == "subject_centric":
                gold = gold_gene
                prompt = build_prompt_subject_centric(q)

            else:
                # Defensive fallback for unexpected modes
                gold = ""
                prompt = f"Query: {q}\n"

            # Call the model using the project wrapper call_model
            try:
                answer = call_model(prompt) or ""
                answer = answer.strip()
            except Exception:
                # Mark parse errors to keep downstream metrics consistent
                answer = "PARSE_ERROR"

            pred = answer if answer else "PARSE_ERROR"

            # Semantic evaluation: compute similarity and acceptance info
            ok, info = semantic_match(pred, gold, task=mode, threshold=sms_threshold)

            # Prefer explicit jw field; fallback to 'score' if jw missing
            jw = info.get("jw", info.get("score", 0.0))
            sms_scores.append(jw)

            gold_all.append(gold)
            pred_all.append(pred)

            # Write a detailed JSONL record for offline inspection
            # Store only sms_jw and matched_synonym to keep outputs compact and consistent
            fout.write(json.dumps({
                "query": q,
                "gold": gold,
                "pred": pred,
                "triple_extracted": triple,
                "matched_synonym": info.get("matched_synonym"),
                "sms": {"jw": jw, "matched_synonym": info.get("matched_synonym")},
                "retrieval_query_sentence": None,
                "retrieval_query_factoid": None,
                "retrieved_sentences": None,
                "retrieved_factoids": None
            }, ensure_ascii=False) + "\n")

    # Compute aggregate metrics and print a concise summary
    metrics = compute_metrics(gold_all, pred_all, sms_scores, sms_threshold=sms_threshold)

    print(f"\n=== {system_name} — METRICS ===")
    print("Exact Match (EM):", metrics["em"])
    print("Semantic Accuracy (SMS ≥ threshold):", metrics["sms"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print()

    return metrics


# ---------------------------------------------------------
# Main entrypoint for quick launcher
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=500)
    parser.add_argument("--sms_threshold", type=float, default=0.80)
    parser.add_argument("--force_rebuild_indices", action="store_true")
    args = parser.parse_args()

    num_queries = args.num_queries
    sms_threshold = args.sms_threshold
    force_rebuild = args.force_rebuild_indices

    # Print execution parameters for traceability
    print("\nExecution parameters:")
    print(f"num_queries = {num_queries}")
    print(f"sms_thresh  = {sms_threshold}\n")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load TBGA train and validation splits
    with open(os.path.normpath(os.path.join(BASE_DIR, "..", "..", "benchmark", "TBGA", "TBGA_train.txt")), encoding="utf-8") as f:
        tbga_train = [json.loads(line) for line in f]

    with open(os.path.normpath(os.path.join(BASE_DIR, "..", "..", "benchmark", "TBGA", "TBGA_val.txt")), encoding="utf-8") as f:
        tbga_val_full = [json.loads(line) for line in f]

    # Filter out invalid examples and limit to requested number
    tbga_val = [x for x in tbga_val_full if x.get("relation", "NA") != "NA"][:num_queries]

    # Optionally ensure indices exist; ignore failures in quick mode
    try:
        if "ensure_indices" in globals():
            ensure_indices(force_rebuild=force_rebuild)
    except Exception:
        pass

    # Load MeSH descriptors and supplementary records
    mesh_desc = load_mesh_dict(os.path.normpath(os.path.join(BASE_DIR, "..", "mesh", "desc2026.xml")))
    mesh_supp = load_mesh_dict(os.path.normpath(os.path.join(BASE_DIR, "..", "mesh", "supp2026.xml")))
    mesh_dict = {**mesh_desc, **mesh_supp}

    # Load HGNC gene list used by the extractor
    gene_path = os.path.normpath(os.path.join(BASE_DIR, "..", "genes", "hgnc_complete_set.txt"))
    gene_dict = load_gene_dict(gene_path)

    # Build known-entity lists used by the extractor
    genes, diseases, relations = load_known_entities(mesh_dict, gene_dict)

    # Generate synthetic queries for the benchmark tasks
    relation_queries = generate_relation_queries_for_items(tbga_val)
    object_queries   = generate_object_queries_for_items(tbga_val)
    subject_queries  = generate_subject_queries_for_items(tbga_val)

    # Run the three no-context evaluations and write JSONL outputs
    evaluate_system(
        tbga_val,
        relation_queries,
        "results/no_context/relation_centric.jsonl",
        "No‑Context Relation-Centric",
        mode="relation_centric",
        sms_threshold=sms_threshold,
        genes=genes,
        diseases=diseases,
        relations=relations
    )

    evaluate_system(
        tbga_val,
        object_queries,
        "results/no_context/object_centric.jsonl",
        "No‑Context Object-Centric",
        mode="object_centric",
        sms_threshold=sms_threshold,
        genes=genes,
        diseases=diseases,
        relations=relations
    )

    evaluate_system(
        tbga_val,
        subject_queries,
        "results/no_context/subject_centric.jsonl",
        "No‑Context Subject-Centric",
        mode="subject_centric",
        sms_threshold=sms_threshold,
        genes=genes,
        diseases=diseases,
        relations=relations
    )
