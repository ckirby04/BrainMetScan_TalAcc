# BrainMetScan — MetRAG

A two-stage pipeline for **brain metastasis analysis** from multi-sequence MRI:

1. **Segmentation** — a lightweight 3D U-Net that localizes metastatic lesions from MRI volumes.
2. **Retrieval-Augmented Generation (RAG)** — extracts case features, retrieves similar prior cases and relevant medical-knowledge facts from a vector database, and generates a clinical-style report.

The project is built around the [BrainMetShare](https://aimi.stanford.edu/brainmetshare) dataset format, which provides four co-registered MRI sequences per case: `t1_pre`, `t1_gd` (gadolinium), `flair`, and `bravo`.

---

## Repository structure

```
1.1/
├── config.yaml                  # Central configuration (paths, model, training, RAG)
├── requirements.txt             # Python dependencies
├── models/                      # Pretrained segmentation weights
│   ├── best_model.pth           #   best validation Dice checkpoint
│   └── final_model.pth          #   final-epoch checkpoint
├── scripts/
│   ├── train_and_demo.py        # End-to-end: train → build RAG DB → run demo case
│   └── visualize_segmentation.py# Sliding-window inference + overlay visualization
├── src/
│   ├── segmentation/
│   │   ├── unet.py              # LightweightUNet3D + CombinedLoss (Dice + BCE)
│   │   ├── dataset.py          # BrainMetDataset, train/val split, 3D patching
│   │   ├── train.py            # Training entry point
│   │   └── inference.py        # Volume inference
│   └── rag/
│       ├── feature_extractor.py# Per-case image embeddings + features
│       ├── build_database.py   # Builds ChromaDB vector DB + medical knowledge base
│       ├── query.py            # Retrieves similar cases/facts, generates report
│       └── add_literature.py   # Adds literature facts to the knowledge base
└── outputs/                     # Predictions, reports, visualizations (gitignored)
```

> **Note:** The MRI dataset itself is **not** included in this repository (it is ~24 GB and lives outside version control). Point the pipeline at your local data via `config.yaml` or the command-line flags below. Expected layout is one directory per case (e.g. `data/train/Mets_005/`) containing the four sequence volumes and a segmentation mask.

---

## Installation

Requires **Python 3.10+** and (recommended) a CUDA-capable GPU.

```bash
# from the 1.1/ directory
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Set `device: "cpu"` in `config.yaml` if you do not have a GPU.

---

## Usage

### Full pipeline (train + RAG + demo)

```bash
python scripts/train_and_demo.py \
    --train_dir data/train \
    --metadata_path data/metadata.csv \
    --num_epochs 100 \
    --device cuda
```

Useful flags:
- `--skip_segmentation_training` — reuse an existing model in the output dir
- `--skip_rag_building` — reuse an existing ChromaDB database
- `--example_case data/train/Mets_005` — case to run through the demo

### Visualize a segmentation

```bash
python scripts/visualize_segmentation.py
```

Runs sliding-window inference (default `96³` window, 0.5 overlap) and saves an overlay of the predicted lesion mask.

### Query the RAG system for a single case

```bash
python src/rag/query.py \
    --case_dir data/train/Mets_005 \
    --db_path outputs/rag/chromadb \
    --output_dir outputs/rag \
    --k_cases 5 --k_facts 3
```

---

## Model

`LightweightUNet3D` — a memory-efficient encoder–decoder 3D U-Net designed for consumer GPUs:

- Double-conv blocks (Conv3d → BatchNorm → ReLU) with dropout
- Configurable `base_channels` (default 16) and `depth` (default 3)
- Trained on `96³` patches with mixed-precision (AMP)
- **Loss:** `CombinedLoss` = 0.7 · Dice + 0.3 · BCE
- **Optimizer:** AdamW with cosine-annealing LR schedule

Inference uses sliding-window aggregation over the full volume.

---

## RAG component

- **Vector store:** ChromaDB (`brain_mets_cases` collection), populated from per-case image embeddings.
- **Knowledge base:** a curated set of clinical facts about brain metastases (epidemiology, imaging characteristics, treatment considerations), extensible via `add_literature.py`.
- **Report generation:** retrieves the *k* most similar cases and *k* most relevant facts. Optionally uses an OpenAI model (`use_openai: true` in `config.yaml`, requires `OPENAI_API_KEY`) for natural-language report synthesis.

---

## Configuration

All defaults live in [`config.yaml`](config.yaml) — data paths, output directories, model architecture, training hyperparameters, and RAG retrieval settings. Command-line flags override the relevant values where supported.

---

## Disclaimer

This is a **research/educational** project. It is **not** a medical device and must **not** be used for clinical diagnosis or treatment decisions.
