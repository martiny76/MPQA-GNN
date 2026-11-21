# MPQA-GNN

Heterogeneous graph attention network on top of a 4-bit QLoRA FLAN-T5 encoder for joint polarity and intensity classification of opinion expressions in MPQA 2.0.

## Method

- **Input:** prompt of the form `aspect=<expression> | sentence=<sentence> | type=<gold type>`.
- **Encoder:** FLAN-T5 (`large`, `xl` or `xxl`), 4-bit NF4 with LoRA adapters; a learned mix of the last four layers.
- **Graph:** `token`, `span`, `span_source` (nested sources) and `target` nodes, with syntax, span, source and target relation families (plus reverse edges), processed by a stack of heterogeneous GATv2 layers.
- **Heads:** span-, source- and target-conditioned readouts per task. Polarity: flat head + hierarchical gate/direction. Intensity: flat head + ordinal head + EMD loss.
- **Post-hoc calibration:** neutral bias and intensity offsets tuned on the validation split only.
- **Evaluation:** 5-fold cross-validation; standard weighted/macro F1 and the boundary-tolerant intensity F1.

Two baselines are included: `encoder_softmax` (encoder + mean pooling + softmax heads) and `text2text` (FLAN-T5 generating the labels).

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A CUDA GPU is required (4-bit quantization). Install `torch` for your platform first if it is not already available.

## Data

The corpus is not included. MPQA 2.0 is subject to its own license; obtain it from the official source. The code expects two preprocessed JSON files in `data/` (or set `DATA_DIR`):

| File | Content |
|---|---|
| `mpqa2_samples.json` | `csds_objects`: list of items with `unique_id`, `text`, `head`, `head_start`, `head_end`, `polarity`, `intensity`, `nested_source`, `target`, `annotation_type` |
| `mpqa2_folds.json` | `IDs_trainset_fold_k`, `IDs_validationset_fold_k`, `IDs_testset_fold_k` for `k = 1..5` |

## Usage

Open `mpqa_gnn.ipynb` and run all cells. Settings are in the Config cell.

- `MODEL_SIZE`: `'large'`, `'xl'` (default) or `'xxl'` (about 12-14 GB VRAM in 4-bit).
- `EXPERIMENT_QUEUE` / `SEED_LIST`: experiments and seeds to run. Alternatively set `HSGAT_EXPERIMENTS="full,no_gnn"` in the environment.
- `FOLDS_TO_RUN`: subset of folds, e.g. `[4]`.

Experiments are defined in `EXPERIMENTS`: `full`, edge-family ablations (`edges_*`), nested-source variants (`src_*`), `no_gnn`, intensity-head ablations (`int_*`), training-component ablations (`no_ema`, `no_span_skip`, ...), and the two baselines (`baseline_softmax`, `baseline_text2text`).

Outputs are written to `OUTPUT_DIR` (default `outputs/`): checkpoints in `checkpoints/` and one `results/<experiment>_seed<seed>.json` per run. After a run, `collect_results_table()` prints the ablation table with paired Wilcoxon tests against `full`.
