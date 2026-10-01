# Granularity RERAG

This module implements the granularity‑aware retrieval and re‑ranking pipeline used inside the broader **GDA‑Extraction** project. It provides proposition‑level and factoid‑level retrieval, entity normalization, synonym expansion, and evaluation utilities.

## Prerequisites

Before installing this module, download the main project:

- [GDA‑Extraction](https://github.com/GDAMining/gda-extraction)

```powershell
git clone https://github.com/GDAMining/gda-extraction.git
```

## Ollama setup

The pipeline calls a local LLM through the Ollama CLI (`ollama run llama3:8b`), so Ollama and the model must be installed on the machine.

1. Download and install Ollama for Windows: [https://ollama.com/download/windows](https://ollama.com/download/windows)
2. Open a new PowerShell window and check that the installation worked:

```powershell
   ollama --version
```

3. Download the model used by the pipeline (about 4.7 GB):

```powershell
   ollama pull llama3:8b
```

4. (Recommended) Run the model once to check that it works and to load it for the first time:

```powershell
   ollama run llama3:8b
```

   Type `/bye` to exit.

**Notes**

- Ollama must be installed and the `ollama` command must be available in your `PATH` before launching the pipeline.
- The model name must be exactly `llama3:8b`, as it is the one referenced in the code.

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
2. **MeSH descriptors and supplementary concepts** (`desc2026.xml` and `supp2026.xml`): [MeSH XML files](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/)

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