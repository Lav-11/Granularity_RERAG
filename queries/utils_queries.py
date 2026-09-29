# utils_queries.py
# Consolidated utilities for MeSH/HGNC processing, retrieval, and triple extraction.
# This module provides:
#  - MeSH parsing and optional precomputed indices (pickle)
#  - HGNC loading and normalization helpers
#  - normalization utilities for tokens and synonyms
#  - a triple extractor that uses dictionary matching with an LLM fallback
#  - retrieval helpers for sentence/factoid FAISS indices
#  - semantic matching utilities and small CLI helper functions
#
# Notes:
#  - Many functions assume UTF-8 encoded input files.
#  - Some operations (embedding, FAISS) require heavy dependencies and GPU for best performance.
#  - The module is designed to be robust to missing precomputed indices: it will attempt to build them if allowed.
#  - Where LLMs are invoked, the code uses the local `ollama` CLI.
#  - This project only evaluates llama3:8b: there is no model-choice branching anywhere
#    in this file anymore (removed on purpose, see call_model).

import warnings
warnings.filterwarnings(
    "ignore",
    message="torch.utils._pytree._register_pytree_node is deprecated"
)

import os
import json
import random
import numpy as np
import subprocess
import re
import gc
import xml.etree.ElementTree as ET
import glob
import pickle
import time
from collections import defaultdict
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModel
from rapidfuzz.distance import JaroWinkler
import faiss

# Deterministic seeds for reproducibility where applicable
random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path for precomputed indices (recommended to build once and reuse)
MESH_INDICES_FILE = os.path.join(BASE_DIR, "mesh_indices.pkl")

# =========================================================
# CONTRIEVER — lazy loading of embedding model
# =========================================================
# The contriever model is loaded lazily to avoid heavy startup cost at import time.
# - load_contriever: returns tokenizer and model, moves model to device and sets eval mode.
# - embed_query: tokenizes and encodes a text query, returns a numpy float32 vector.
EMB_MODEL_NAME = "facebook/contriever"
_emb_tokenizer = None
_emb_model = None
_emb_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_contriever():
    """
    Lazy-load the Contriever tokenizer and model.
    Returns:
        (tokenizer, model)
    Side effects:
        - loads model weights into memory and moves to _emb_device
        - sets model to eval mode
    """
    global _emb_tokenizer, _emb_model
    if _emb_model is None:
        _emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
        _emb_model = AutoModel.from_pretrained(EMB_MODEL_NAME)
        _emb_model.eval()
        _emb_model.to(_emb_device)
    return _emb_tokenizer, _emb_model

def embed_query(text):
    """
    Embed a text query using the Contriever model.
    Args:
        text (str): input text to embed
    Returns:
        numpy.ndarray: 2D array with shape (1, hidden_size) dtype float32
    Notes:
        - This function moves tensors to the configured device and returns a CPU numpy array.
        - It attempts to free GPU memory after use.
    """
    tok, model = load_contriever()
    inputs = tok(text, padding=True, truncation=True, return_tensors="pt").to(_emb_device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
    del inputs, outputs
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except:
        pass
    return emb.cpu().numpy().astype("float32")


# =========================================================
# NORMALIZATION (genes and relation synonyms)
# =========================================================
# These helpers standardize gene symbols and relation synonyms for robust matching.

def normalize_gene(gene):
    """
    Normalize a gene symbol for dictionary matching.
    - Lowercases, strips whitespace, and removes spaces, hyphens and underscores.
    Returns None if input is None.
    """
    if gene is None:
        return None
    gene = gene.strip().lower()
    gene = gene.replace(" ", "").replace("-", "").replace("_", "")
    return gene


# Canonical relation keys map to lists of synonyms (some use underscores).
RELATION_SYNONYMS = {

    # genomic alteration: any change in genetic information
    "genomic_alterations": ["mutation", "mutated", "genetic_change"],

    # biomarker: a measurable indicator of some biological state or condition
    "biomarker": ["associated", "linked_to", "correlated_with"],

    # therapeutic: of or relating to the treatment of disease or disorders by remedial agents or methods
    "therapeutic": ["treats", "used_for", "therapy_for"]
}

def normalize_tokens(text):
    """
    Tokenize and normalize a text string for matching.
    - Converts to lowercase
    - Replaces underscores and commas with spaces
    - Removes punctuation (except word characters and whitespace)
    - Collapses multiple spaces
    Returns:
        list[str] tokens
    """
    text = text.lower()
    text = re.sub(r"[_\,]", " ", text)      # underscores and commas -> space
    text = re.sub(r"[^\w\s]", "", text)     # remove other punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def normalize_with_synonyms(rel, syns):
    """
    Map a relation string to its canonical key using a synonyms dict.
    If rel matches a canonical key or any synonym (case-insensitive), return the canonical key.
    Otherwise return the original rel unchanged.
    """
    if rel is None:
        return None
    r_low = rel.lower().strip()
    for canonical, lst in syns.items():
        if r_low == canonical.lower():
            return canonical
        for s in lst:
            if r_low == s.lower():
                return canonical
    return rel


# =========================================================
# LLM CALL
# =========================================================
# Wrapper around the local `ollama` CLI. This wrapper:
#  - sanitizes prompts (removes problematic markers)
#  - retries on garbage output
#  - returns "PARSE_ERROR" on persistent failure
#
# Important: using the CLI limits control over generation parameters (temperature, stop, etc.).
# Consider switching to an API client if you need deterministic generation or fine-grained parameters.

def _is_garbage_output(s, max_repeat_ratio=0.6, min_alnum_ratio=0.1):
    """
    Heuristic to detect degenerate outputs from the model.

    Returns True when the output is likely unusable, for example when:
      - the string is empty or only whitespace
      - a single character dominates the output beyond max_repeat_ratio
      - the fraction of alphanumeric characters is below min_alnum_ratio
      - there are long repeated non-alphanumeric sequences such as @@@@@@@
    These checks are intentionally conservative to avoid discarding valid short answers.
    """
    # If the string is empty or only whitespace, treat it as garbage immediately
    if not s or not s.strip():
        return True

    # Work on the trimmed string to avoid counting leading/trailing whitespace
    chars = list(s.strip())

    # Count occurrences of the most frequent character
    # If one character composes more than max_repeat_ratio of the output,
    # the output is likely a repetition or degenerate token stream
    most_common = max((chars.count(c) for c in set(chars)), default=0)
    if most_common / max(1, len(chars)) > max_repeat_ratio:
        return True

    # Compute the fraction of alphanumeric characters
    # Very low alphanumeric ratio suggests the output is mostly punctuation or control characters
    alnum = sum(1 for c in chars if c.isalnum())
    if alnum / max(1, len(chars)) < min_alnum_ratio:
        return True

    # Detect long repeated non-alphanumeric sequences using a regex
    # This catches patterns like @@@@@@@ which are common in degenerate outputs
    if re.search(r'([^\w\s])\1{6,}', s):
        return True

    # If none of the heuristics triggered, consider the output non-garbage
    return False


def call_model(prompt, max_retries=2, retry_delay=0.5):
    """
    Robust wrapper to call the local Ollama CLI for short-answer generation.

    Always uses llama3:8b. There is no model-choice parameter anymore: this
    project only ever evaluates llama3:8b, so the previous "llama" vs
    "deepseek-r1:14b" branching was removed rather than kept as dead code.

    Behavior:
      - removes fake system markers like <<SYS>>
      - prepends a short system preamble instructing the model to return a single short token
      - retries a few times if the output looks like garbage
      - returns "PARSE_ERROR" if no valid output is obtained
    Args:
      prompt (str): the textual prompt to send to the model
      max_retries (int): number of retries on garbage output
      retry_delay (float): seconds to wait between retries
    Returns:
      str: model output or "PARSE_ERROR"
    """
    cleaned_prompt = prompt.replace("<<SYS>>", "").strip()
    system_preamble = (
        "You MUST output only the requested short answer (no extra text).\n"
        "Return a single short token or 'PARSE_ERROR' if you cannot answer.\n\n"
    )
    full_prompt = system_preamble + cleaned_prompt

    cmd = ["ollama", "run", "llama3:8b"]

    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        try:
            proc = subprocess.run(
                cmd,
                input=full_prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=25
            )
            out = proc.stdout.strip()
        except subprocess.TimeoutExpired:
            out = ""

        # Minimal cleanup: remove code fences and leading labels
        out = out.replace("```", "").strip()
        prefixes = ["Gene:", "Disease:", "Relation:", "Answer:", "Final:", "-", "*"]
        for p in prefixes:
            if out.lower().startswith(p.lower()):
                out = out[len(p):].strip()

        # If output looks like garbage, retry with a stricter instruction
        if _is_garbage_output(out):
            if attempt > max_retries:
                return "PARSE_ERROR"
            full_prompt = (
                "You MUST output a single short token only. If unsure, output PARSE_ERROR.\n"
                + cleaned_prompt
            )
            time.sleep(retry_delay)
            continue

        return out if out else "PARSE_ERROR"

    return "PARSE_ERROR"


# =========================================================
# LLM RAW CALL FOR TRIPLE EXTRACTOR
# =========================================================
# This function is used by the triple extractor to get JSON-like outputs from the model.
# It is intentionally simpler than call_model because the triple extractor performs
# additional parsing and validation of the returned text.
def ollama_generate(prompt):
    """
    Run the Ollama CLI (llama3:8b) with the given prompt and return raw stdout.
    This function intentionally returns raw text for downstream parsing.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3:8b"],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=30
        )
        return result.stdout.strip()
    except:
        return ""


# =========================================================
# MESH NORMALIZATION UTILITIES
# =========================================================
# Utilities to normalize MeSH strings for indexing and matching.
# - umls_norm: canonical lowercased form with punctuation removed (except commas)
# - generate_mesh_variants: produce simple variants (reversed comma forms, token-sorted forms)
@lru_cache(maxsize=200000)
def umls_norm(text):
    """
    Normalize a MeSH string for dictionary keys.
    - Lowercases
    - Removes characters that are not word, whitespace or comma
    - Collapses whitespace
    Returns empty string for None input.
    """
    if text is None:
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s,]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def generate_mesh_variants(term):
    """
    Generate a small set of normalized variants for a MeSH term.
    Variants include:
      - normalized term
      - reversed comma-separated form (if present)
      - token-joined and token-sorted forms for multi-token terms
    These variants help match different surface forms encountered in XML.
    """
    term = umls_norm(term)
    variants = set()
    variants.add(term)
    if "," in term:
        parts = [p.strip() for p in term.split(",")]
        variants.add(" ".join(parts[::-1]))
    tokens = term.replace(",", " ").split()
    if len(tokens) > 1:
        variants.add(" ".join(tokens))
        variants.add(" ".join(sorted(tokens)))
    return list(variants)


# =========================================================
# INDEX BUILD / LOAD (one-time precompute recommended)
# =========================================================
# build_mesh_indices: parse MeSH XML files and build three indices:
#   - norm_index: normalized string -> list of occurrences (file, text, norm)
#   - disease_norm_index: normalized disease string -> list of entries (mesh_id, origin, text, norm)
#   - token_index: token -> list of disease_norm strings (for fast candidate lookup)
#
# The function serializes the indices to a pickle file for reuse.
def build_mesh_indices(mesh_paths=None, out_file=MESH_INDICES_FILE):
    """
    Parse MeSH XML files and build indices for fast lookup.
    Args:
      mesh_paths (list[str] or None): list of XML file paths; if None, glob ../mesh/*.xml
      out_file (str): path to write the pickle file
    Returns:
      dict with keys norm_index, disease_norm_index, token_index
    Notes:
      - This is an expensive one-time operation; prefer to reuse the pickled indices.
      - token_index values are converted to lists for serialization.
    """
    if mesh_paths is None:
        mesh_paths = glob.glob(os.path.join(BASE_DIR, "../mesh/*.xml"))

    norm_index = defaultdict(list)
    disease_norm_index = defaultdict(list)
    token_index = defaultdict(set)

    for p in mesh_paths:
        try:
            tree = ET.parse(p)
            root = tree.getroot()
        except Exception:
            # Skip files that fail to parse; caller may want to log or sanitize them.
            continue

        # Collect all <String> elements (generic strings in MeSH XML)
        for elem in root.findall(".//String"):
            txt = elem.text
            if txt and txt.strip():
                n = umls_norm(txt.strip())
                norm_index[n].append({"file": p, "text": txt.strip(), "norm": n})

        # DescriptorRecord entries (main descriptors)
        for rec in root.findall(".//DescriptorRecord"):
            mesh_id = rec.findtext("DescriptorUI") or ""
            dn = rec.findtext(".//DescriptorName/String")
            if dn and dn.strip():
                n = umls_norm(dn.strip())
                entry = {"mesh_id": mesh_id, "origin": "descriptor_name", "text": dn.strip(), "norm": n}
                disease_norm_index[n].append(entry)
                for t in n.split():
                    token_index[t].add(n)
            for term in rec.findall(".//Concept/TermList/Term/String"):
                if term is not None and term.text and term.text.strip():
                    ttxt = term.text.strip()
                    n = umls_norm(ttxt)
                    entry = {"mesh_id": mesh_id, "origin": "descriptor_concept_term", "text": ttxt, "norm": n}
                    disease_norm_index[n].append(entry)
                    for t in n.split():
                        token_index[t].add(n)

        # SupplementaryConceptRecord entries (SCR)
        for rec in root.findall(".//SupplementaryConceptRecord"):
            mesh_id = rec.findtext("SupplementalRecordUI") or ""
            main = rec.findtext(".//ConceptList/Concept/TermList/Term/String")
            if main and main.strip():
                n = umls_norm(main.strip())
                entry = {"mesh_id": mesh_id, "origin": "supplementary_main", "text": main.strip(), "norm": n}
                disease_norm_index[n].append(entry)
                for t in n.split():
                    token_index[t].add(n)
            for term in rec.findall(".//Concept/TermList/Term/String"):
                if term is not None and term.text and term.text.strip():
                    ttxt = term.text.strip()
                    n = umls_norm(ttxt)
                    entry = {"mesh_id": mesh_id, "origin": "supplementary_term", "text": ttxt, "norm": n}
                    disease_norm_index[n].append(entry)
                    for t in n.split():
                        token_index[t].add(n)

    # Convert token_index sets to lists for pickling
    token_index = {k: list(v) for k, v in token_index.items()}
    indices = {
        "norm_index": dict(norm_index),
        "disease_norm_index": dict(disease_norm_index),
        "token_index": token_index
    }

    with open(out_file, "wb") as f:
        pickle.dump(indices, f, protocol=pickle.HIGHEST_PROTOCOL)

    return indices


def load_mesh_indices(path=MESH_INDICES_FILE):
    """
    Load precomputed indices from disk.
    Raises an exception if the file cannot be opened or is invalid.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# Attempt to load indices at import time for a fast path. If unavailable, INDICES remains None.
try:
    INDICES = load_mesh_indices()
except Exception:
    INDICES = None


# =========================================================
# LOAD MESH XML + HGNC TSV
# =========================================================
# These functions parse a single MeSH XML file into a compact dictionary and
# load HGNC TSV into a gene dictionary suitable for local matching.

def load_mesh_dict(path):
    """
    Load a single MeSH XML file into a dictionary mapping mesh_id -> {name, synonyms}.
    - DescriptorRecord entries populate descriptor names and synonyms.
    - SupplementaryConceptRecord entries populate SCR names and synonyms.
    Returns:
      dict: mesh_id -> {"name": name_variant, "synonyms": [variants...]}
    Notes:
      - The function uses generate_mesh_variants to produce normalized synonyms.
      - If the XML is malformed, ET.parse will raise; callers should handle or sanitize files.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    mesh_dict = {}

    # DescriptorRecord (main descriptors)
    for record in root.findall(".//DescriptorRecord"):
        mesh_id = record.findtext("DescriptorUI")

        name_raw = record.findtext(".//DescriptorName/String")
        name_variants = generate_mesh_variants(name_raw) if name_raw else [""]
        name = name_variants[0] if name_variants else (umls_norm(name_raw) if name_raw else "")

        synonyms = set()
        for term in record.findall(".//Concept/TermList/Term/String"): # finds all synonyms 
            t_raw = term.text
            if t_raw and t_raw.strip():
                for v in generate_mesh_variants(t_raw):
                    synonyms.add(v)

        for term in record.findall(".//Concept/Term/String"):
            t_raw = term.text
            if t_raw and t_raw.strip():
                for v in generate_mesh_variants(t_raw):
                    synonyms.add(v)

        if name_raw and name_raw.strip():
            synonyms.add(umls_norm(name_raw))

        mesh_dict[mesh_id] = {
            "name": name,
            "synonyms": sorted(synonyms)
        }

    # SupplementaryConceptRecord (SCR)
    for record in root.findall(".//SupplementaryConceptRecord"):
        mesh_id = record.findtext("SupplementalRecordUI")

        name_raw = record.findtext(".//ConceptList/Concept/TermList/Term/String")
        if not name_raw:
            # fallback: try any <String> under the SCR
            any_str = record.find(".//String")
            name_raw = any_str.text if any_str is not None and any_str.text else None

        if not name_raw or not name_raw.strip():
            continue

        name_variants = generate_mesh_variants(name_raw)
        name = name_variants[0] if name_variants else umls_norm(name_raw)

        synonyms = set()
        for term in record.findall(".//Concept/TermList/Term/String"):
            t_raw = term.text
            if t_raw and t_raw.strip():
                for v in generate_mesh_variants(t_raw):
                    synonyms.add(v)

        for term in record.findall(".//Term/String"):
            t_raw = term.text
            if t_raw and t_raw.strip():
                for v in generate_mesh_variants(t_raw):
                    synonyms.add(v)

        synonyms.add(umls_norm(name_raw))

        mesh_dict[mesh_id] = {
            "name": name,
            "synonyms": sorted(synonyms)
        }

    return mesh_dict


def load_gene_dict(path):
    """
    Load HGNC TSV file into a dictionary mapping hgnc_id -> {symbol, synonyms}.
    Expected TSV header contains columns: hgnc_id, symbol, alias_symbol.
    Normalization:
      - symbol is normalized via normalize_gene
      - alias_symbol is split by comma and normalized; short or numeric aliases are discarded
    Returns:
      dict: hgnc_id -> {"symbol": normalized_symbol, "synonyms": [normalized_syns]}
    """
    gene_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")

        idx_id = header.index("hgnc_id")
        idx_symbol = header.index("symbol")
        idx_synonyms = header.index("alias_symbol")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(idx_id, idx_symbol, idx_synonyms):
                continue

            hgnc_id = parts[idx_id]
            symbol = normalize_gene(parts[idx_symbol])
            syns_raw = parts[idx_synonyms]

            synonyms = []
            if syns_raw:
                for s in syns_raw.split(","):
                    s_norm = normalize_gene(s)

                    # Heuristics to filter out noisy aliases:
                    if len(s_norm) < 4:
                        continue
                    if s_norm in symbol:
                        continue
                    if symbol in s_norm:
                        continue
                    if s_norm.isdigit():
                        continue

                    synonyms.append(s_norm)

            gene_dict[hgnc_id] = {
                "symbol": symbol,
                "synonyms": synonyms
            }

    return gene_dict


# =========================================================
# SINGLE-TOKEN STATS BUILDER
# =========================================================
# Build statistics about single-token MeSH entries to help disambiguation.
SINGLE_TOKEN_COUNTS = None        # token_norm -> count
SINGLE_TOKEN_IS_DESCRIPTOR = None # set of token_norm that are descriptor single-token

def build_single_token_stats(mesh_dict):
    """
    Count occurrences of single-token normalized strings across MeSH entries.
    Returns:
      (counts, descriptor_set)
      - counts: dict token_norm -> frequency
      - descriptor_set: set of tokens that appear as descriptor names (not only synonyms)
    Use:
      - helps decide whether a single-token match is ambiguous or likely a descriptor.
    """
    counts = {}
    descriptor_set = set()

    for mesh_id, info in mesh_dict.items():
        name = info.get("name", "")
        if name:
            n = umls_norm(name)
            if len(n.split()) == 1:
                counts[n] = counts.get(n, 0) + 1
                descriptor_set.add(n)

        for s in info.get("synonyms", []):
            if not s:
                continue
            s_norm = umls_norm(s)
            if len(s_norm.split()) == 1:
                counts[s_norm] = counts.get(s_norm, 0) + 1

    return counts, descriptor_set


# =========================================================
# BUILD VOCAB FROM MESH + HGNC
# =========================================================
def load_known_entities(mesh_dict, gene_dict):
    """
    Build sorted lists of known genes, diseases and relations from loaded dictionaries.
    Also populates SINGLE_TOKEN_COUNTS and SINGLE_TOKEN_IS_DESCRIPTOR for downstream heuristics.
    Returns:
      (genes, diseases, relations) as sorted lists of strings.
    """
    global SINGLE_TOKEN_COUNTS, SINGLE_TOKEN_IS_DESCRIPTOR

    genes = set()
    diseases = set()
    relations = set(RELATION_SYNONYMS.keys())

    for g in gene_dict.values():
        genes.add(g["symbol"])
        for syn in g.get("synonyms", []):
            genes.add(syn)

    for d in mesh_dict.values():
        diseases.add(d["name"])
        for syn in d.get("synonyms", []):
            diseases.add(syn)

    try:
        SINGLE_TOKEN_COUNTS, SINGLE_TOKEN_IS_DESCRIPTOR = build_single_token_stats(mesh_dict)
    except Exception:
        SINGLE_TOKEN_COUNTS, SINGLE_TOKEN_IS_DESCRIPTOR = {}, set()

    return sorted(genes), sorted(diseases), sorted(relations)


# =========================================================
# TRIPLE EXTRACTOR — final version (uses indices if available)
# =========================================================
# This component extracts (gene, relation, disease) triples from a question.
# Strategy:
#  - Use dictionary matching (HGNC + MeSH indices) for fields that are expected to be present
#  - Never extract the task target field (the field the downstream system must retrieve)
#  - Fall back to LLM only for fields that dictionary matching failed to find
#
# The extractor returns:
#  - gene, relation, disease (strings or None)
#  - llm_used (bool)
#  - llm_raw_output (raw text from the model, for debugging)
#  - source: dict indicating whether each field came from dictionary or llm
#
# Important: match_disease uses a coverage heuristic to avoid spurious matches (see its docstring).
_TEMPLATE_STOPWORDS = None

def _get_template_stopwords():
    """
    Build (once) the set of structural words that appear in the question templates.
    These words are removed when checking whether a candidate disease explains the entire question.
    The templates (RELATION_CENTRIC_TEMPLATES, etc.) must be defined elsewhere in the codebase.
    """
    global _TEMPLATE_STOPWORDS
    if _TEMPLATE_STOPWORDS is None:
        words = set()
        all_templates = (
            RELATION_CENTRIC_TEMPLATES
            + OBJECT_CENTRIC_TEMPLATES
            + SUBJECT_CENTRIC_TEMPLATES
        )
        for t in all_templates:
            skeleton = (
                t.replace("{gene}", " ")
                 .replace("{relation}", " ")
                 .replace("{disease}", " ")
            )
            words.update(normalize_tokens(skeleton))
        _TEMPLATE_STOPWORDS = words
    return _TEMPLATE_STOPWORDS


def match_disease(q, exclude_tokens=None):
    """
    Attempt to find a disease mention inside the question q using the precomputed MeSH indices.

    Strategy (current implementation)
      - Normalize the question and extract tokens.
      - Use token_index to find candidate normalized disease keys that share tokens with q.
      - For each candidate (starting from the longest / most specific), check whether
        the candidate's tokens are all present in the question tokens.
      - Coverage check: after removing candidate tokens, exclude_tokens (e.g., gene/relation)
        and template stopwords, the candidate must explain the remaining content of the question.

    Returns
      disease text (original MeSH text) or None if no reliable match is found.

    Notes and current limitations (important)
      - **Template dependency and conservative coverage**: the function relies on a precomputed
        set of structural words (`_get_template_stopwords()`) derived from a fixed set of question
        templates. The coverage check is intentionally conservative: it accepts a candidate only
        when, after removing candidate tokens, exclude_tokens and template stopwords, there is
        nothing left to explain in the question. This reduces false positives but **can produce
        false negatives** on free‑form or rephrased questions that contain legitimate extra words
        (age ranges, populations, subordinate clauses, politeness phrases, etc.).
      - **False negatives on non‑template phrasing**: any additional content not present in the
        template skeleton remains as "leftover" and may cause a correct candidate to be rejected.
        In practice this affects recall on natural, unconstrained user queries.
      - **Single‑token ambiguity**: single‑word matches are treated conservatively to avoid
        spurious matches (see SINGLE_TOKEN_COUNTS / SINGLE_TOKEN_IS_DESCRIPTOR). This reduces
        false positives but can further lower recall for short disease names.

    Implementation note
      - If INDICES are not loaded, the function will attempt to load or build them (controlled by
        environment variables). The function is conservative by design to avoid false positives
        on short tokens; this tradeoff is documented above and should be revisited if higher
        recall on free‑form questions is required.
    """

     # Quick exit on null input
    if q is None:
        return None

    # Normalize question and split into tokens
    q_norm = umls_norm(q)
    q_tokens = normalize_tokens(q_norm)
    if not q_tokens:
        return None
    q_tokens_set = set(q_tokens)

    # Prepare exclude tokens (e.g., gene or relation already matched)
    exclude_tokens = set(exclude_tokens) if exclude_tokens else set()
    # Load the set of structural words from templates to ignore them in coverage check
    template_stopwords = _get_template_stopwords()

    # Ensure INDICES are available; try load then optionally build
    global INDICES
    if INDICES is None:
        try:
            INDICES = load_mesh_indices(MESH_INDICES_FILE)
        except Exception:
            INDICES = None
        if INDICES is None and os.environ.get("UTILS_AUTO_BUILD_INDICES", "1") == "1":
            try:
                INDICES = build_mesh_indices(mesh_paths=None, out_file=MESH_INDICES_FILE)
            except Exception as e:
                print(f"DEBUG: build_mesh_indices failed: {e}")

    # If still missing, cannot match
    if INDICES is None:
        return None

    disease_norm_index = INDICES.get("disease_norm_index", {})
    token_index = INDICES.get("token_index", {})

    # Candidate selection: collect normalized keys that share at least one token with the question
    candidate_norms = set()
    for t in q_tokens_set:
        for norm_key in token_index.get(t, []):
            candidate_norms.add(norm_key)

    # Filter candidates: keep only those whose tokens all appear in the question
    candidates = []
    for norm_key in candidate_norms:
        nk_tokens = normalize_tokens(norm_key)
        if not nk_tokens:
            continue
        # require that every token of the normalized candidate is present in the question
        if all(t in q_tokens_set for t in nk_tokens):
            candidates.append((nk_tokens, norm_key))

    # Prefer more specific candidates (more tokens) first
    candidates.sort(key=lambda c: len(c[0]), reverse=True)

    # Coverage check: after removing candidate tokens, exclude_tokens and template stopwords,
    # there should be nothing left to explain in the question
    for nk_tokens, norm_key in candidates:
        leftover = q_tokens_set - set(nk_tokens) - exclude_tokens - template_stopwords
        if not leftover:
            entries = disease_norm_index.get(norm_key, [])
            if entries:
                # return the original MeSH text for the first matching entry
                return entries[0].get("text", norm_key)

    # No reliable match found
    return None


def extract_triplet_from_question(question, genes, diseases, relations, task):
    """
    Extract (gene, relation, disease) from a question.
    - Uses local dictionaries (HGNC + MeSH indices) first.
    - Falls back to LLM only for fields that the dictionary did not find and that are expected to be present.
    - The 'target' field for the task is never extracted here (it is what the downstream system must retrieve).
    Args:
      question (str): the input question text
      genes (list[str]): known gene symbols (from load_known_entities)
      diseases (list[str]): known disease names (from load_known_entities)
      relations (list[str]): known relation canonical keys (from load_known_entities)
      task (str): one of "relation_centric", "subject_centric", "object_centric"
    Returns:
      dict with keys: gene, relation, disease, llm_used, llm_raw_output, source
    """
    llm_used = False
    llm_raw_output = None

    gene = None
    relation = None
    disease = None

    source = {
        "gene": "none",
        "relation": "none",
        "disease": "none"
    }

    q = question.lower()

    TARGET_FIELD = {
        "relation_centric": "relation",
        "subject_centric": "gene",
        "object_centric": "disease",
    }
    target = TARGET_FIELD.get(task)

    # -----------------------------
    # 1. MATCH GENE (HGNC)
    # -----------------------------
    def match_gene_local():
        """
        Match a gene symbol inside the question using token-level membership.
        The gene list contains normalized symbols and synonyms; normalize both sides for robust matching.
        """
        q_tokens = set(normalize_tokens(q))
        for g in genes:
            g_norm = normalize_gene(g)
            if g_norm in q_tokens:
                return g
        return None

    # -----------------------------
    # 2. MATCH RELATION (canonical + synonyms)
    # -----------------------------
    def match_relation_local():
        """
        Match a relation by checking whether the canonical key or any synonym appears as a substring in the question.
        Note: this is a simple heuristic and may miss matches that require token normalization (underscores vs spaces).
        """
        for r in relations:
            if r.lower() in q:
                return r
            for syn in RELATION_SYNONYMS.get(r, []):
                if syn.lower() in q:
                    return r
        return None

    # -----------------------------
    # 3. APPLY DICTIONARY MATCHING
    #    Never attempt to extract the task target field (it is not present in the question by design).
    # -----------------------------
    if target != "gene":
        gene = match_gene_local()
        if gene:
            source["gene"] = "dictionary"

    if target != "relation":
        relation = match_relation_local()
        if relation:
            source["relation"] = "dictionary"

    if target != "disease":
        exclude = set()
        if gene:
            exclude |= set(normalize_tokens(gene))
        if relation:
            exclude |= set(normalize_tokens(relation))
        disease = match_disease(q, exclude_tokens=exclude)
        if disease:
            source["disease"] = "dictionary"

    # -----------------------------
    # 4. LLM FALLBACK
    #    Only for the known fields of the current task (never for the target),
    #    and only for fields that dictionary matching failed to find.
    # -----------------------------
    def llm_extract(fields):
        """
        Ask the LLM to extract the requested fields and return a dict mapping field->value or None.
        The prompt enforces strict JSON-only output; the function robustly extracts the first JSON object
        from the model output using raw_decode to avoid issues with trailing text.
        """
        nonlocal llm_used, llm_raw_output

        if not fields:
            return {}

        llm_used = True

        template = "{\n" + ",\n".join([f'  "{f}": "VALUE"' for f in fields]) + "\n}"

        prompt = (
            "You MUST output ONLY valid JSON.\n"
            f"{template}\n\n"
            "STRICT RULES:\n"
            "- Each VALUE must be a short name only (a gene symbol, a disease name, or a relation type) — never a sentence.\n"
            "- DO NOT use acronyms or short forms for diseases. Always expand diseases in full (e.g., use 'Hepatocellular Carcinoma' not 'HCC').\n"
            "- Use full words, avoid underscores, and prefer canonical disease names.\n"
            "- NO verbs, NO explanations, NO extra text outside the JSON.\n\n"
            f"Question: {question}\n"
        )

        raw = ollama_generate(prompt)
        llm_raw_output = raw

        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

        # Find the first JSON object and parse only that object.
        start = cleaned.find("{")
        if start == -1:
            return {f: None for f in fields}

        try:
            data, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except Exception:
            return {f: None for f in fields}

        result = {}
        for f in fields:
            val = data.get(f)
            result[f] = val.strip() if isinstance(val, str) and val.strip() else None

        return result

    missing_fields = []
    if target != "gene" and gene is None:
        missing_fields.append("gene")
    if target != "relation" and relation is None:
        missing_fields.append("relation")
    if target != "disease" and disease is None:
        missing_fields.append("disease")

    if missing_fields:
        data = llm_extract(missing_fields)

        if data.get("gene") is not None:
            gene = data["gene"]
            source["gene"] = "llm"

        if data.get("relation") is not None:
            relation = data["relation"]
            source["relation"] = "llm"

        if data.get("disease") is not None:
            disease = data["disease"]
            source["disease"] = "llm"

    return {
        "gene": gene,
        "relation": relation,
        "disease": disease,
        "llm_used": llm_used,
        "llm_raw_output": llm_raw_output,
        "source": source
    }


# =========================================================
# RETRIEVAL
# =========================================================
# Helpers to load precomputed sentence/factoid embeddings and FAISS indices,
# and to retrieve top-k results with semantic deduplication.

def load_sentence_retriever(emb_dir=None):
    """
    Load sentence-level FAISS retriever.
    If emb_dir is provided, load index + metadata from that directory.
    Otherwise fall back to the default location.
    """
    if emb_dir is None:
        emb_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "sentence", "data", "embeddings_sentence"))

    emb_path = os.path.join(emb_dir, "sentence_embeddings.npy")
    index_path = os.path.join(emb_dir, "index.faiss")
    meta_path = os.path.join(emb_dir, "metadata.jsonl")

    if not (os.path.exists(emb_path) and os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Sentence retriever files not found in {emb_dir}")

    embeddings = np.load(emb_path)

    index = faiss.read_index(index_path)

    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    return embeddings, index, metadata



def load_factoid_retriever(emb_dir=None):
    """
    Load factoid-level FAISS retriever.
    If emb_dir is provided, load index + metadata from that directory.
    Otherwise fall back to the default location.
    """
    if emb_dir is None:
        emb_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "factoid", "data", "embeddings_factoid"))

    emb_path = os.path.join(emb_dir, "embeddings.npy")
    index_path = os.path.join(emb_dir, "index.faiss")
    meta_path = os.path.join(emb_dir, "metadata.jsonl")

    if not (os.path.exists(emb_path) and os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Factoid retriever files not found in {emb_dir}")

    embeddings = np.load(emb_path)

    index = faiss.read_index(index_path)

    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    return embeddings, index, metadata



def retrieve_top_k(query_embedding, index, metadata, k, oversample_factor=4):
    """
    Retrieve top-k hits from a FAISS index with simple textual deduplication.

    Signature preserved: retrieve_top_k(query_embedding, index, metadata, k, oversample_factor=4)

    Behavior:
      - If index is None or search fails, returns [].
      - Requests top_n = k * oversample_factor candidates from FAISS to allow removing
        duplicate textual hits while still returning up to k unique results.
      - Deduplicates by normalized text (prefers 'sentence' key for sentence metadata,
        otherwise builds a compact factoid string from gene/relation/disease).
      - Returns a list of metadata objects (same type as entries in `metadata`).
    """
    # Defensive early exit
    if index is None:
        return []

    # Ensure oversample_factor is sensible
    try:
        oversample_factor = int(oversample_factor)
        if oversample_factor < 1:
            oversample_factor = 1
    except Exception:
        oversample_factor = 4

    top_n = max(k * oversample_factor, k + 5)

    # Ensure query_embedding is float32 2D
    try:
        qv = query_embedding.astype("float32")
    except Exception:
        qv = query_embedding

    # Run FAISS search; guard against exceptions
    try:
        D, I = index.search(qv, int(top_n))
    except Exception:
        return []

    # Normalize index result shape for single-query case
    if hasattr(I, "shape") and I.shape[0] == 1:
        idxs = I[0]
    else:
        try:
            idxs = I[0]
        except Exception:
            # fallback: try to iterate I as a flat list
            try:
                idxs = list(I)
            except Exception:
                return []

    results = []
    seen_texts = set()

    for idx in idxs:
        # FAISS may return -1 for padding; skip invalid indices
        try:
            idx_int = int(idx)
        except Exception:
            continue
        if idx_int < 0:
            continue

        # Safely fetch metadata entry
        try:
            meta = metadata[idx_int]
        except Exception:
            # If metadata lookup fails, skip this hit
            continue

        # Extract canonical text for deduplication:
        if isinstance(meta, dict):
            # Prefer 'sentence' key for sentence metadata
            text = meta.get("sentence") or " ".join(filter(None, [meta.get("gene",""), meta.get("relation",""), meta.get("disease","")])).strip()
        else:
            text = str(meta)

        # Normalize whitespace for stable dedupe
        text_norm = " ".join(text.split()).strip()

        if text_norm in seen_texts:
            continue

        seen_texts.add(text_norm)
        results.append(meta)

        if len(results) >= k:
            break

    return results



# =========================================================
# RETRIEVAL QUERY BUILDERS
# =========================================================
# Small helpers to build natural-language queries for retrieval components.

def build_sentence_retrieval_query(gene, relation, disease, mode):
    """
    Build a short sentence retrieval query using only the provided fields.
    Side effect: if a global list named `retrieved_sentences` exists, it will be deduplicated in place.
    Returns: str or None
    """
    include_map = {
        "subject_centric": ["relation", "disease"],
        "object_centric": ["gene", "relation"],
        "relation_centric": ["gene", "disease"],
    }

    mode_key = (mode or "").lower()
    parts = []
    for key in include_map.get(mode_key, []):
        if key == "gene" and gene:
            parts.append(str(gene).strip())
        elif key == "relation" and relation:
            parts.append(str(relation).strip())
        elif key == "disease" and disease:
            parts.append(str(disease).strip())

    # deduplicate parts while preserving order
    seen = set()
    cleaned = []
    for p in parts:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        cleaned.append(p)

    query = " ".join(cleaned) if cleaned else None

    # Automatic in-place deduplication of global `retrieved_sentences` if present
    try:
        raw = globals().get("retrieved_sentences", None)
        if isinstance(raw, list):
            out = []
            seen_snips = set()
            for s in raw:
                if not s:
                    continue
                key = re.sub(r"<[^>]+>", "", s)
                key = re.sub(r"\s+", " ", key).strip().lower()
                if not key or key in seen_snips:
                    continue
                seen_snips.add(key)
                out.append(s)
            globals()["retrieved_sentences"] = out
    except Exception:
        pass

    return query


def build_factoid_retrieval_query(gene, relation, disease, mode):
    """
    Build a factoid retrieval query using only the provided fields.
    Side effect: if a global list named `retrieved_factoids` exists, it will be deduplicated in place.
    Returns: str or None
    """
    include_map = {
        "subject_centric": ["relation", "disease"],
        "object_centric": ["gene", "relation"],
        "relation_centric": ["gene", "disease"],
    }

    mode_key = (mode or "").lower()
    parts = []
    for key in include_map.get(mode_key, []):
        if key == "gene" and gene:
            parts.append(str(gene).strip())
        elif key == "relation" and relation:
            parts.append(str(relation).strip())
        elif key == "disease" and disease:
            parts.append(str(disease).strip())

    # deduplicate parts while preserving order
    seen = set()
    cleaned = []
    for p in parts:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        cleaned.append(p)

    query = " ".join(cleaned) if cleaned else None

    # Automatic in-place deduplication of global `retrieved_factoids` if present
    try:
        raw = globals().get("retrieved_factoids", None)
        if isinstance(raw, list):
            out = []
            seen_snips = set()
            for s in raw:
                if not s:
                    continue
                key = re.sub(r"<[^>]+>", "", s)
                key = re.sub(r"\s+", " ", key).strip().lower()
                if not key or key in seen_snips:
                    continue
                seen_snips.add(key)
                out.append(s)
            globals()["retrieved_factoids"] = out
    except Exception:
        pass

    return query




# =========================================================
# QUERY GENERATION
# =========================================================

RELATION_CENTRIC_TEMPLATES = [
    "What is the relationship between {gene} and {disease}?",
    "Determine the relation between {gene} and {disease}.",
    "What type of relation links {gene} and {disease}?"
]

OBJECT_CENTRIC_TEMPLATES = [
    "Which disease is {relation} to {gene}?",
    "Which disease does {gene} show {relation} with?",
    "What disease is linked to {gene} through {relation}?"
]

SUBJECT_CENTRIC_TEMPLATES = [
    "Which gene acts as {relation} for {disease}?",
    "Which gene shows {relation} in {disease}?",
    "Identify the gene that has a {relation} role in {disease}."
]

def generate_relation_queries_for_items(tbga_items):
    return [
        random.choice(RELATION_CENTRIC_TEMPLATES).format(
            gene=item["h"]["name"], disease=item["t"]["name"]
        )
        for item in tbga_items
    ]

def generate_object_queries_for_items(tbga_items):
    return [
        random.choice(OBJECT_CENTRIC_TEMPLATES).format(
            gene=item["h"]["name"], relation=item["relation"]
        )
        for item in tbga_items
    ]

def generate_subject_queries_for_items(tbga_items):
    return [
        random.choice(SUBJECT_CENTRIC_TEMPLATES).format(
            disease=item["t"]["name"], relation=item["relation"]
        )
        for item in tbga_items
    ]


# =========================================================
# SEMANTIC MATCH
# =========================================================

_DISEASE_SYNONYM_GROUPS = None  # mesh_id -> set of texts (name + synonyms)


def _get_disease_synonym_groups():
    """
    Group all texts that share the same mesh_id from INDICES['disease_norm_index'].
    This expands object_centric gold answers with real MeSH synonyms instead of
    comparing pred only against the literal gold string.
    """
    global _DISEASE_SYNONYM_GROUPS, INDICES

    if _DISEASE_SYNONYM_GROUPS is not None:
        return _DISEASE_SYNONYM_GROUPS

    if INDICES is None:
        try:
            INDICES = load_mesh_indices(MESH_INDICES_FILE)
        except Exception:
            INDICES = None
        if INDICES is None and os.environ.get("UTILS_AUTO_BUILD_INDICES", "1") == "1":
            try:
                INDICES = build_mesh_indices(mesh_paths=None, out_file=MESH_INDICES_FILE)
            except Exception as e:
                print(f"DEBUG: build_mesh_indices failed: {e}")

    groups = defaultdict(set)
    if INDICES is not None:
        for entries in INDICES.get("disease_norm_index", {}).values():
            for e in entries:
                mesh_id = e.get("mesh_id")
                text = e.get("text")
                if mesh_id and text:
                    groups[mesh_id].add(text)

    _DISEASE_SYNONYM_GROUPS = dict(groups)
    return _DISEASE_SYNONYM_GROUPS


def _get_disease_synonyms(gold):
    """Return known MeSH synonyms for the gold (excluding the gold itself)."""
    groups = _get_disease_synonym_groups()
    disease_norm_index = (INDICES or {}).get("disease_norm_index", {})

    gold_norm = umls_norm(gold)
    entries = disease_norm_index.get(gold_norm, [])

    syns = set()
    for e in entries:
        mesh_id = e.get("mesh_id")
        if mesh_id and mesh_id in groups:
            syns |= groups[mesh_id]

    syns.discard(gold)
    return sorted(syns)


def _get_gene_synonyms(gold, gene_dict):
    """
    Return known HGNC synonyms for the gold (excluding the gold itself).
    Requires gene_dict produced by load_gene_dict(path).
    If gene_dict is not provided, return an empty list.
    """
    if not gene_dict:
        return []

    gold_norm = normalize_gene(gold)
    for info in gene_dict.values():
        if info.get("symbol") == gold_norm or gold_norm in info.get("synonyms", []):
            group = set(info.get("synonyms", []))
            group.add(info.get("symbol"))
            group.discard(gold_norm)
            return sorted(group)

    return []


def _looks_like_spurious_prefix_match(a, b):
    """
    Detect pathological case where a short generic token (e.g., "gene") is a literal
    prefix of a much longer phrase (e.g., "genetic change") producing artificially
    high Jaro-Winkler scores. Only applies when the longer string is multi-word.
    """
    shorter, longer = sorted([a, b], key=len)
    if not shorter or not longer:
        return False
    if len(longer.split()) <= 1:
        return False
    length_ratio = len(shorter) / len(longer)
    return longer.startswith(shorter) and length_ratio < 0.5


def semantic_match(pred, gold, task=None, threshold=0.80, gene_dict=None):
    """
    Compare pred against gold using Jaro-Winkler only for relation/object,
    and exact canonical match for subject (genes).
    Returns (ok: bool, info: dict) with keys "jw", "score", "matched_synonym".
    """
    # Defensive normalization of inputs
    pred_raw = "" if pred is None else str(pred).strip()
    gold_raw = "" if gold is None else str(gold).strip()

    pred_clean = pred_raw.lower().replace("_", " ")
    gold_clean = gold_raw.lower().replace("_", " ")

    # SUBJECT / GENE: exact canonical match only
    if task == "subject_centric":
        pred_norm = normalize_gene(pred_raw) or ""
        gold_norm = normalize_gene(gold_raw) or ""

        # Build local reverse map from gene_dict if provided
        gene_map = None
        if gene_dict:
            try:
                gm = {}
                for info in gene_dict.values():
                    sym = info.get("symbol")
                    if sym:
                        sym_norm = normalize_gene(sym)
                        gm[sym_norm] = sym_norm
                    for s in info.get("synonyms", []):
                        s_norm = normalize_gene(s)
                        if s_norm:
                            gm[s_norm] = normalize_gene(info.get("symbol") or s_norm)
                gene_map = gm
            except Exception:
                gene_map = None

        # Prefer global GENE_REV_MAP if present
        if "GENE_REV_MAP" in globals() and globals().get("GENE_REV_MAP"):
            gene_map = globals().get("GENE_REV_MAP")

        pred_canon = gene_map.get(pred_norm, pred_norm) if gene_map else pred_norm
        gold_canon = gold_norm

        ok = (pred_canon == gold_canon)
        jw = 1.0 if ok else 0.0
        matched = pred_raw if ok and pred_norm != gold_canon else None

        return ok, {
            "jw": float(jw),
            "score": float(jw),
            "matched_synonym": matched
        }

    # RELATION / OBJECT: synonym expansion + JW only
    gold_syns = [gold_raw]

    if task == "object_centric":
        if "_get_disease_synonyms" in globals() and callable(globals().get("_get_disease_synonyms")):
            try:
                extra = _get_disease_synonyms(gold_raw)
                if extra:
                    gold_syns = [gold_raw] + extra
            except Exception:
                gold_syns = [gold_raw]
    elif task == "relation_centric":
        try:
            rel_list = RELATION_SYNONYMS.get(gold_raw, [])
            gold_syns = [gold_raw] + rel_list
        except Exception:
            gold_syns = [gold_raw]
    else:
        gold_syns = [gold_raw]

    best_score = 0.0
    best_syn = None

    for gs in gold_syns:
        gs_clean = (gs or "").lower().replace("_", " ")

        try:
            jw = JaroWinkler.normalized_similarity(pred_clean, gs_clean)
        except Exception:
            try:
                jw = JaroWinkler.similarity(pred_clean, gs_clean)
            except Exception:
                jw = 1.0 if pred_clean == gs_clean else 0.0

        score = float(jw)

        if score > best_score:
            best_score = score
            best_syn = gs

    ok = best_score >= threshold

    return ok, {
        "jw": float(best_score),
        "score": float(best_score),
        "matched_synonym": best_syn if ok else None
    }




# =========================================================
# METRICS
# =========================================================

def compute_metrics(gold_all, pred_all, sms_scores, sms_threshold=0.80):
    """
    Compute simple evaluation metrics:
      - EM (exact match) between pred and gold (case-insensitive)
      - SMS: fraction of examples with sms_scores >= sms_threshold
      - precision and recall derived from SMS counts (approximate for this use-case)
    Returns:
      dict with keys: em, sms, precision, recall
    """
    if len(gold_all) == 0:
        return {"em": 0.0, "sms": 0.0, "precision": 0.0, "recall": 0.0}

    em_correct = sum(int(p.strip().lower() == g.strip().lower())
                     for p, g in zip(pred_all, gold_all))
    em = em_correct / len(gold_all)

    sms_correct = sum(int(s >= sms_threshold) for s in sms_scores)
    sms = sms_correct / len(gold_all)

    TP = sms_correct
    FP = len(gold_all) - sms_correct
    FN = FP

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    return {"em": em, "sms": sms, "precision": precision, "recall": recall}


# =========================================================
# JSONL LOADING
# =========================================================

def load_results_jsonl(path):
    """
    Load a JSONL file where each line is a JSON object.
    Uses errors='replace' to avoid crashes on malformed characters.
    Returns:
      list of parsed JSON objects
    """
    data = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# =========================================================
# Helper CLI utilities
# =========================================================

def ensure_indices(mesh_paths=None, out_file=MESH_INDICES_FILE, force_rebuild=False):
    """
    Ensure that MeSH indices are available. If not present or force_rebuild is True,
    build the indices and return them.
    Returns:
      loaded or newly built indices dict
    """
    global INDICES
    if not force_rebuild and INDICES is not None:
        return INDICES
    if not force_rebuild and os.path.exists(out_file):
        try:
            INDICES = load_mesh_indices(out_file)
            return INDICES
        except Exception:
            pass
    indices = build_mesh_indices(mesh_paths=mesh_paths, out_file=out_file)
    INDICES = indices
    return INDICES


if __name__ == "__main__":
    # Quick smoke test to verify module import
    print("utils_queries loaded")