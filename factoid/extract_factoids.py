import os
import json

# ---------------------------------------------------------
# Paths and I/O configuration
# ---------------------------------------------------------
# SCRIPT_DIR: directory where this script lives
# RAW_DIR: path to the original TBGA benchmark files (train/val/test)
# DATA_DIR: local data folder where extracted factoid JSONL files will be written
# OUT_FACTOID: output folder for per-split factoid JSONL files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "benchmark", "TBGA"))

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_FACTOID = os.path.join(DATA_DIR, "jsonl_factoid")
os.makedirs(OUT_FACTOID, exist_ok=True)

# Expected input TBGA files (JSONL with fields like 'h','t','relation','text')
FILES = {
    "train": os.path.join(RAW_DIR, "TBGA_train.txt"),
    "val":   os.path.join(RAW_DIR, "TBGA_val.txt"),
    "test":  os.path.join(RAW_DIR, "TBGA_test.txt"),
}


# ---------------------------------------------------------
# Factoid extraction
# ---------------------------------------------------------
def extract_factoids(input_path, output_path):
    """
    Extract factoid-level records from a TBGA JSONL file.

    Behavior and expectations:
      - Reads the input TBGA file line-by-line (each line is a JSON object).
      - For each entry, extracts gene (h.name), disease (t.name), relation and sentence (text).
      - Excludes entries where relation == "NA" (these are not factoids).
      - Writes one JSON object per line to output_path with keys:
        'gene', 'relation', 'disease', 'factoid', 'sentence'.
      - Returns (total_lines_read, extracted_count) for basic reporting.

    Notes:
      - The factoid string is built as: "<gene> <relation> <disease>".
      - The function preserves the original sentence text as-is (no normalization).
      - Use this output for building factoid indices or for factoid-level retrieval experiments.
    """
    total = 0
    extracted = 0
    skipped_na = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            data = json.loads(line)

            # Extract expected fields from TBGA format
            gene = data["h"]["name"]
            disease = data["t"]["name"]
            relation = data["relation"]
            sentence = data["text"]

            # Exclude entries with NA relations (not factoids)
            if relation == "NA":
                skipped_na += 1
                continue

            # Canonical factoid string used by downstream indexing/evaluation
            factoid_str = f"{gene} {relation} {disease}"

            obj = {
                "gene": gene,
                "relation": relation,
                "disease": disease,
                "factoid": factoid_str,
                "sentence": sentence
            }

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            extracted += 1

    print(f"Skipped NA factoids: {skipped_na}")
    return total, extracted


# ---------------------------------------------------------
# Main: process train/val/test splits
# ---------------------------------------------------------
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        out_path = os.path.join(OUT_FACTOID, f"tbga_{split}_factoids.jsonl")

        print(f"\nProcessing split: {split.upper()}")
        total, extracted = extract_factoids(FILES[split], out_path)

        print(f"Total entries: {total}")
        print(f"Extracted factoids (non-NA): {extracted}")
        print(f"Output written to: {out_path}")
