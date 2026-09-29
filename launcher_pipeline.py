"""
launcher_pipeline.py

Orchestrator for the Granularity Rerag evaluation pipeline.

This script runs the full end-to-end evaluation workflow:
  1. Sentence extraction (builds sentence metadata)
  2. Sentence embeddings (builds sentence-level FAISS index)
  3. Factoid extraction (builds factoid metadata)
  4. Factoid embeddings (builds factoid-level FAISS index)
  5. No-Context evaluation (LLM-only baseline)
  6. Sentence-RAG evaluation (retrieval-augmented generation using sentence index)
  7. Factoid-RAG evaluation (retrieval-augmented generation using factoid index)
  8. Aggregate reporting and qualitative dumps (JSON and image summary)

Design goals and notes:
  - Keep each step idempotent: if artifacts already exist, skip rebuilding.
  - Use subprocess calls to run per-stage scripts located in subfolders.
  - Produce JSONL per-system outputs and consolidated qualitative dumps for manual inspection.
  - Provide robust JSONL loading and defensive checks to avoid crashes on malformed files.
  - This file focuses on orchestration and reporting; core logic (retrieval, embedding,
    triple extraction, semantic matching) lives in the `queries` and `utils_queries` modules.
"""

import os
import subprocess
import json
import shutil
import matplotlib.pyplot as plt
import sys
import re

try:
    # ---------------------------------------------------------------------
    # Base paths and import setup
    # ---------------------------------------------------------------------
    # BASE_DIR: root directory for this pipeline (directory containing this file)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Ensure the queries package is importable regardless of current working dir.
    # This allows subprocesses and imports inside `queries/` to resolve consistently.
    QUERIES_DIR_REAL = os.path.join(os.path.dirname(__file__), "queries")
    if QUERIES_DIR_REAL not in sys.path:
        sys.path.insert(0, QUERIES_DIR_REAL)

    # Import compute_metrics from utils_queries for metric aggregation.
    # We intentionally keep a local JSONL loader below to handle malformed lines robustly.
    from utils_queries import compute_metrics

    # ---------------------------------------------------------------------
    # Directory layout used by the pipeline
    # ---------------------------------------------------------------------
    SENTENCE_DIR = os.path.join(BASE_DIR, "sentence")
    FACTOID_DIR = os.path.join(BASE_DIR, "factoid")
    QUERIES_DIR = os.path.join(BASE_DIR, "queries")
    RESULTS_DIR = os.path.join(QUERIES_DIR, "results")

    # ---------------------------------------------------------------------
    # Configuration (tunable)
    # ---------------------------------------------------------------------
    # NUM_QUERIES: how many synthetic queries to generate / evaluate per run
    # SMS_THRESHOLD: semantic-match threshold used to split positive/negative examples
    NUM_QUERIES = 5
    SMS_THRESHOLD = 0.86

    # ---------------------------------------------------------------------
    # Small utility wrappers
    # ---------------------------------------------------------------------
    def run(cmd, cwd=None):
        """
        Run a subprocess command and print it for traceability.

        Args:
            cmd (list[str]): command and arguments to execute.
            cwd (str|None): working directory for the subprocess.
        """
        print(f"\n>>> RUNNING: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=cwd, shell=False, check=True)

    def file_exists(path):
        """Return True if path exists and is a regular file."""
        return os.path.exists(path) and os.path.isfile(path)

    def load_results_jsonl(path):
        """
        Robust JSONL loader.

        - Skips empty lines.
        - Prints a warning for malformed JSON lines but continues processing.
        - Returns a list of parsed JSON objects (possibly empty).
        """
        data = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except Exception as e:
                        # Warn but do not raise: we want the pipeline to continue.
                        print(f"WARNING: failed to parse line {i} in {path}: {e}")
        except FileNotFoundError:
            # Upstream steps may not have produced the file; caller will handle empty lists.
            print(f"WARNING: results file not found: {path}")
        return data

    # ---------------------------------------------------------------------
    # Prepare results directories (clean run)
    # ---------------------------------------------------------------------
    # Remove previous results to ensure a fresh evaluation run.
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    # Create per-system result directories
    os.makedirs(os.path.join(RESULTS_DIR, "no_context"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "sentence_rag"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "factoid_rag"), exist_ok=True)

    # ---------------------------------------------------------------------
    # Paths for embeddings and metadata (expected outputs of earlier steps)
    # ---------------------------------------------------------------------
    SENT_EMB = os.path.join(SENTENCE_DIR, "data", "embeddings_sentence", "sentence_embeddings.npy")
    SENT_INDEX = os.path.join(SENTENCE_DIR, "data", "embeddings_sentence", "index.faiss")
    SENT_META = os.path.join(SENTENCE_DIR, "data", "embeddings_sentence", "metadata.jsonl")

    FACT_EMB = os.path.join(FACTOID_DIR, "data", "embeddings_factoid", "embeddings.npy")
    FACT_INDEX = os.path.join(FACTOID_DIR, "data", "embeddings_factoid", "index.faiss")
    FACT_META = os.path.join(FACTOID_DIR, "data", "embeddings_factoid", "metadata.jsonl")

    # ---------------------------------------------------------------------
    # STEP 1 — Sentence extraction
    # - If sentence metadata does not exist, run the extraction script.
    # - This step produces the sentence metadata JSONL used by the sentence retriever.
    # ---------------------------------------------------------------------
    print("\n=== STEP 1: Sentence extraction ===")
    if not file_exists(SENT_META):
        run([sys.executable, "extract_sentences.py"], cwd=SENTENCE_DIR)
    else:
        print("Sentence metadata already exists. Skipping.")

    # ---------------------------------------------------------------------
    # STEP 2 — Sentence embeddings
    # - Build sentence embeddings and FAISS index if missing.
    # - This step is expensive and typically run once; subsequent runs reuse artifacts.
    # ---------------------------------------------------------------------
    print("\n=== STEP 2: Sentence embeddings ===")
    if not (file_exists(SENT_EMB) and file_exists(SENT_INDEX)):
        run([sys.executable, "build_sentence_embeddings.py"], cwd=SENTENCE_DIR)
    else:
        print("Sentence embeddings already exist. Skipping.")

    # ---------------------------------------------------------------------
    # STEP 3 — Factoid extraction
    # - Extract structured factoids (gene–relation–disease triples) from the corpus.
    # - Produces metadata used by the factoid retriever.
    # ---------------------------------------------------------------------
    print("\n=== STEP 3: Factoid extraction ===")
    if not file_exists(FACT_META):
        run([sys.executable, "extract_factoids.py"], cwd=FACTOID_DIR)
    else:
        print("Factoid metadata already exists. Skipping.")

    # ---------------------------------------------------------------------
    # STEP 4 — Factoid embeddings
    # - Build factoid embeddings and FAISS index if missing.
    # ---------------------------------------------------------------------
    print("\n=== STEP 4: Factoid embeddings ===")
    if not (file_exists(FACT_EMB) and file_exists(FACT_INDEX)):
        run([sys.executable, "build_factoid_embeddings.py"], cwd=FACTOID_DIR)
    else:
        print("Factoid embeddings already exist. Skipping.")

    # ---------------------------------------------------------------------
    # STEP 5 — Run No‑Context system
    # - Baseline evaluation: LLM answers without retrieval augmentation.
    # - We pass only the minimal CLI arguments required by the runner.
    # ---------------------------------------------------------------------
    print("\n=== STEP 5: Running No‑Context ===")
    run([
        sys.executable,
        "run_no_context.py",
        "--num_queries", str(NUM_QUERIES),
        "--sms_threshold", str(SMS_THRESHOLD)
    ], cwd=QUERIES_DIR)

    # ---------------------------------------------------------------------
    # STEP 6 — Run Sentence‑RAG system
    # - Retrieval-augmented generation using the sentence-level FAISS index.
    # - We explicitly pass the embeddings directory so the runner can load the correct index.
    # ---------------------------------------------------------------------
    print("\n=== STEP 6: Running Sentence‑RAG ===")
    run([
        sys.executable,
        "run_sentence_rag.py",
        "--num_queries", str(NUM_QUERIES),
        "--sms_threshold", str(SMS_THRESHOLD),
    ], cwd=QUERIES_DIR)

    # ---------------------------------------------------------------------
    # STEP 7 — Run Factoid‑RAG system
    # - Retrieval-augmented generation using the factoid-level FAISS index.
    # ---------------------------------------------------------------------
    print("\n=== STEP 7: Running Factoid‑RAG ===")
    run([
        sys.executable,
        "run_factoid_rag.py",
        "--num_queries", str(NUM_QUERIES),
        "--sms_threshold", str(SMS_THRESHOLD),
    ], cwd=QUERIES_DIR)

    print("\n=== Pipeline completed successfully ===")

    # ---------------------------------------------------------------------
    # STEP 8 — Build final JSON report
    # - Aggregate EM and average SMS for each system/task into final_report.json
    # - compute_summary is defensive: returns zeros for empty or malformed files.
    # ---------------------------------------------------------------------
    print("\n=== STEP 8: Building final report ===")

    REPORT_PATH = os.path.join(BASE_DIR, "final_report.json")

    def compute_summary(path):
        """
        Compute Exact Match (EM) and average SMS (using sms_jw) for a single JSONL results file.

        Returns (em, sms) as floats. If the file is empty or missing, returns (0.0, 0.0).
        """
        data = load_results_jsonl(path)
        if not data:
            return 0.0, 0.0
        em = sum(1 for x in data if x.get("pred") == x.get("gold")) / len(data)
        sms = sum(x.get("sms", {}).get("jw", 0.0) for x in data) / len(data)
        return em, sms

    report = []

    # Walk the results directory and compute metrics for each JSONL file found.
    for root, dirs, files in os.walk(RESULTS_DIR):
        for file in files:
            if not file.endswith(".jsonl"):
                continue

            full_path = os.path.join(root, file)
            system_type = os.path.basename(root)
            task_name = file.replace(".jsonl", "")
            system_name = f"{system_type}_{task_name}"

            # Skip any "seen" variants if present (legacy behavior)
            if "seen" in system_name:
                continue

            em, sms = compute_summary(full_path)

            report.append({
                "system": system_name,
                "em": em,
                "sms": sms
            })

    # Persist the aggregated report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\nFinal report saved to: {REPORT_PATH}")
    print("\n=== Report generated successfully ===")

    # ---------------------------------------------------------------------
    # STEP 8B — Full qualitative dump
    # - Collects every example from all systems into a single JSON file for manual inspection.
    # - We sanitize values to ensure JSON serializability and to avoid crashes on unexpected types.
    # ---------------------------------------------------------------------
    print("\n=== STEP 8B: Building FULL qualitative dump ===")

    # Debugging output to help diagnose missing result files
    print("DEBUG: RESULTS_DIR exists:", os.path.exists(RESULTS_DIR))
    for sys_name in ["no_context", "sentence_rag", "factoid_rag"]:
        p = os.path.join(RESULTS_DIR, sys_name)
        print(f"DEBUG: {sys_name} path: {p} exists: {os.path.exists(p)}")
        if os.path.exists(p):
            print("  files:", os.listdir(p))

    qualitative = []

    # Iterate systems and tasks and collect per-example metadata
    for sys_name in ["no_context", "sentence_rag", "factoid_rag"]:
        for task in ["relation_centric", "object_centric", "subject_centric"]:

            path = os.path.join(RESULTS_DIR, sys_name, f"{task}.jsonl")

            if not os.path.exists(path):
                # Missing files are expected if a system did not run; log and continue.
                print(f"DEBUG: missing results file: {path}")
                continue

            data = load_results_jsonl(path)

            for entry in data:
                qualitative.append({
                    "system": sys_name,
                    "task": task,
                    "query": entry.get("query", ""),
                    "gold": entry.get("gold", ""),
                    "pred": entry.get("pred", ""),
                    "sms_jw": entry.get("sms", {}).get("jw", 0.0),
                    "matched_synonym": entry.get("sms", {}).get("matched_synonym", None),
                    "triple_extracted": entry.get("triple_extracted", None),
                    "retrieval_query_sentence": entry.get("retrieval_query_sentence", None),
                    "retrieval_query_factoid": entry.get("retrieval_query_factoid", None),
                    "retrieved_sentences": entry.get("retrieved_sentences", None),
                    "retrieved_factoids": entry.get("retrieved_factoids", None)
                })

    # Ensure all retrieved fields are JSON-serializable strings/lists
    for ex in qualitative:
        if ex.get("retrieved_sentences") is None:
            ex["retrieved_sentences"] = []
        else:
            ex["retrieved_sentences"] = [str(s) for s in ex["retrieved_sentences"]]
        if ex.get("retrieved_factoids") is None:
            ex["retrieved_factoids"] = []
        else:
            ex["retrieved_factoids"] = [str(s) for s in ex["retrieved_factoids"]]

    # --- dedupe retrieved lists before saving full qualitative dump ---
    def _dedupe_list(lst):
        seen = set()
        out = []
        for s in lst or []:
            if not s:
                continue
            key = re.sub(r"<[^>]+>", "", s)
            key = re.sub(r"\s+", " ", key).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    for ex in qualitative:
        ex["retrieved_sentences"] = _dedupe_list(ex.get("retrieved_sentences", []))
        ex["retrieved_factoids"] = _dedupe_list(ex.get("retrieved_factoids", []))
    # -------------------------------------------------------------------

    QUAL_FULL = os.path.join(BASE_DIR, "qualitative_dump_full.json")
    with open(QUAL_FULL, "w", encoding="utf-8") as f:
        json.dump(qualitative, f, indent=4, ensure_ascii=False)

    print(f"Full qualitative dump saved to: {QUAL_FULL}\n")

    # ---------------------------------------------------------------------
    # STEP 8C — Positive / Negative dumps
    # - Split the qualitative dump by SMS threshold for quick triage.
    # ---------------------------------------------------------------------
    positive = [x for x in qualitative if x["sms_jw"] >= SMS_THRESHOLD]
    negative = [x for x in qualitative if x["sms_jw"] < SMS_THRESHOLD]

    POS_PATH = os.path.join(BASE_DIR, "qualitative_positive.json")
    NEG_PATH = os.path.join(BASE_DIR, "qualitative_negative.json")

    with open(POS_PATH, "w", encoding="utf-8") as f:
        json.dump(positive, f, indent=4, ensure_ascii=False)

    with open(NEG_PATH, "w", encoding="utf-8") as f:
        json.dump(negative, f, indent=4, ensure_ascii=False)

    print(f"Positive dump saved to: {POS_PATH}")
    print(f"Negative dump saved to: {NEG_PATH}\n")

    # ---------------------------------------------------------------------
    # STEP 9 — Compute SMS metrics table
    # - Build a compact dictionary of SMS scores for each system/task.
    # - Defensive handling for missing or empty result files.
    # ---------------------------------------------------------------------
    print("\n=== STEP 9: Computing SMS metrics ===")

    systems = ["no_context", "sentence_rag", "factoid_rag"]
    tasks = ["relation_centric", "object_centric", "subject_centric"]

    sms_table = {}

    for sys_name in systems:
        for task in tasks:
            key = f"{sys_name}_{task}"
            path = os.path.join(RESULTS_DIR, sys_name, f"{task}.jsonl")

            if not os.path.exists(path):
                continue

            data = load_results_jsonl(path)
            if not data:
                sms_table[key] = 0.0
                continue

            gold = [x.get("gold", "") for x in data]
            pred = [x.get("pred", "") for x in data]
            sms_scores = [x.get("sms", {}).get("jw", 0.0) for x in data]

            metrics = compute_metrics(gold, pred, sms_scores, sms_threshold=SMS_THRESHOLD)
            sms_table[key] = metrics["sms"]

    # ---------------------------------------------------------------------
    # STEP 10 — Print SMS table
    # - Human-readable console summary of SMS per system/task.
    # ---------------------------------------------------------------------
    print("\n=== STEP 10: SMS Table ===")
    print(f"{'SYSTEM':<15} {'REL':<10} {'OBJ':<10} {'SUB':<10}")

    for sys_name in systems:
        rel = sms_table.get(f"{sys_name}_relation_centric", 0.0)
        obj = sms_table.get(f"{sys_name}_object_centric", 0.0)
        sub = sms_table.get(f"{sys_name}_subject_centric", 0.0)

        print(f"{sys_name:<15} {rel:<10.3f} {obj:<10.3f} {sub:<10.3f}")

    # ---------------------------------------------------------------------
    # STEP 11 — Matplotlib table
    # - Save a small PNG summarizing SMS scores for quick visual inspection.
    # ---------------------------------------------------------------------
    print("\n=== STEP 11: Generating Matplotlib Table ===")

    fig, ax = plt.subplots(figsize=(8, 4))

    rows = ["No‑Context", "Sentence‑RAG", "Factoid‑RAG"]
    cols = ["Relation", "Object", "Subject"]

    data = [
        [
            sms_table.get("no_context_relation_centric", 0.0),
            sms_table.get("no_context_object_centric", 0.0),
            sms_table.get("no_context_subject_centric", 0.0)
        ],
        [
            sms_table.get("sentence_rag_relation_centric", 0.0),
            sms_table.get("sentence_rag_object_centric", 0.0),
            sms_table.get("sentence_rag_subject_centric", 0.0)
        ],
        [
            sms_table.get("factoid_rag_relation_centric", 0.0),
            sms_table.get("factoid_rag_object_centric", 0.0),
            sms_table.get("factoid_rag_subject_centric", 0.0)
        ]
    ]

    table = ax.table(
        cellText=[[f"{v:.3f}" for v in row] for row in data],
        rowLabels=rows,
        colLabels=cols,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    ax.axis("off")

    output_path = os.path.join(BASE_DIR, "results_table.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Matplotlib table saved to: {output_path}")
    print("\n=== All done ===")

except KeyboardInterrupt:
    # Graceful shutdown on manual interruption
    print("\n\n>>> Manual interruption detected (CTRL+C).")
    print(">>> Pipeline stopped immediately.\n")
    sys.exit(1)
