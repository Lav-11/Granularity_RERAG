# Granularity RERAG

This module implements the granularity‑aware retrieval and re‑ranking pipeline used inside the broader **GDA‑Extraction** project. It provides proposition‑level and factoid‑level retrieval, entity normalization, synonym expansion, and evaluation utilities.

## Prerequisites

Before installing this module, download the main project:

- [GDA‑Extraction](https://github.com/GDAMining/gda-extraction)

```powershell
git clone https://github.com/GDAMining/gda-extraction.git
```

## Project structure

Place the entire `granularity_rerag` directory inside the main GDA‑Extraction project:

```text
gda-extraction/
│
├── granularity_rerag/
│   ├── factoid/
│   │   ├── build_factoid_embeddings.py
│   │   ├── evaluate_factoid_retriever.py
│   │   └── extract_factoids.py
│   ├── sentence/
│   │   ├── build_sentence_embeddings.py
│   │   ├── evaluate_sentence_retriever.py
│   │   └── extract_sentences.py
│   └── ...
│
└── other_modules/
```

## Data download

Download the following resources before running the pipeline:

1. **HGNC complete set**: [hgnc_complete_set.txt](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt)
This has to go inside the genes folder inside granularity_rerag.

2. **MeSH descriptors and supplementary concepts** (`desc2026.xml` and `supp2026.xml`): [MeSH XML files](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/)
This has to go inside the mesh folder inside granularity_rerag.

## Installation

### Requirements

- Python 3.10 recommended.
- If you want GPU support, CUDA 12.1 is required for the `+cu121` PyTorch wheels included in `requirements.txt`.

### One‑line install (uses both PyPI and PyTorch index)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 -r .\requirements.txt
```

**Notes**

- This installs exactly the versions listed in `requirements.txt` (including `torch==2.2.2+cu121`).
- If the target machine does not have CUDA 12.1, see the CPU alternative below.

### CPU alternative

If users do not have CUDA 12.1, provide them `requirements-cpu.txt`. Create it locally with:

```powershell
Get-Content .\requirements.txt | ForEach-Object { $_ -replace '\+cu[0-9]+','+cpu' } | Set-Content .\requirements-cpu.txt
```

Then install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r .\requirements-cpu.txt
```

## How to run

Inside the `granularity_rerag` folder:

```powershell
python .\launcher_pipeline.py
```