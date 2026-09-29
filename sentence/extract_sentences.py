import os
import json

# ---------------------------------------------------------
# Paths and I/O configuration
# ---------------------------------------------------------
# SCRIPT_DIR: directory where this script lives
# RAW_DIR: path to the original TBGA benchmark files (train/val/test)
# DATA_DIR: local data folder where extracted sentence JSONL files will be written
# OUT_SENT: output folder for per-split sentence JSONL files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "benchmark", "TBGA"))

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_SENT = os.path.join(DATA_DIR, "jsonl_sentences")
os.makedirs(OUT_SENT, exist_ok=True)

# Expected input files (TBGA original files)
FILES = {
    "train": os.path.join(RAW_DIR, "TBGA_train.txt"),
    "val":   os.path.join(RAW_DIR, "TBGA_val.txt"),
    "test":  os.path.join(RAW_DIR, "TBGA_test.txt"),
}


# ---------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------
def extract_sentences(input_path, output_path):
    """
    Extract sentence-level records from a TBGA JSONL file.

    Behavior and expectations:
      - Reads the input TBGA file line-by-line (each line is a JSON object).
      - For each entry, extracts the sentence text and the annotated triple fields:
        'h' (head/gene), 't' (tail/disease) and 'relation'.
      - Writes one JSON object per line to the output_path with keys:
        'sentence', 'gene', 'relation', 'disease'.
      - Returns (total_lines_read, lines_written) for basic reporting.

    Notes:
      - The function does not filter out entries with relation == "NA".
        Keeping NA entries is intentional for downstream RE/RAG experiments that
        may require negative or no-relation examples.
      - The function preserves the original sentence text as-is (no normalization).
    """
    total = 0
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            data = json.loads(line)

            # Extract fields expected in TBGA format
            gene = data["h"]["name"]
            disease = data["t"]["name"]
            relation = data["relation"]
            sentence = data["text"]

            # Compose a compact record for retrieval/indexing pipelines
            obj = {
                "sentence": sentence,
                "gene": gene,
                "relation": relation,
                "disease": disease
            }

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    return total, count


# ---------------------------------------------------------
# Main: process train/val/test splits
# ---------------------------------------------------------
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        out_path = os.path.join(OUT_SENT, f"tbga_{split}_sentences.jsonl")

        print(f"\nProcessing split: {split.upper()}")
        total, extracted = extract_sentences(FILES[split], out_path)

        print(f"Total entries: {total}")
        print(f"Extracted sentences: {extracted}")
        print(f"Output written to: {out_path}")
