# Paired-Unit steering

This repository implements the method proposed by the paper "Beyond Global Contrasts: Paired-Unit Activation Steering for Context-Aware Protein Sequence Design".

| Program | Usage |
| --- | --- |
| `extract_steering_vectors.py` | Build and save a PU vector with optional ASPO/ablation controls. |
| `optimize_sequences.py` | Apply a saved vector, writing sequences and complete paths. |
| `lds_independent.py` | Sample independent LDS pools, score paths, and choose top candidates. |

## 1. Construct and save a PU vector

```bash
python3 extract_steering_vectors.py \
  --construction data/my_construction.csv \
  --pairing-layer 35 \
  --weight-temperature 0.10 \
  --output runs/pu_vectors.npz
```

`--pairing-layer` selects the layer for cosine matching. `Paired-Unit` is the default method; `--method ASPO` builds a centroid baseline with the same script. Saved vectors include all 36 layers and metadata: method, model, construction-file hash, class counts, and matching settings.

## 2. Optimize target proteins

Use the saved vector on any target CSV:

```bash
python3 optimize_sequences.py \
  --vectors runs/pu_vectors.npz --targets data/my_targets.csv \
  --optimal-layer 25 --rounds 4 --sites-per-round 2 \
  --alpha 1.0 --temperature 0 --output-dir runs/pu
```

The optimizer ranks sites by ascending cosine similarity at `--optimal-layer`, masks unused sites, and predicts new amino acids with all-layer steering; the original amino acid is excluded from replacement.

The manuscript settings are in `configs/study_protocol.json`.

## 3. Rank candidates by LDS with an independent pool

Use lds_independent.py to rank optimized proteins by how well their realized sequence changes agree with property-improving directions from a separate labeled reference set.

```bash
python3 lds_independent.py \
  --pool data/held_out_pool.csv \
  --trajectories runs/pu/trajectory_steps.csv \
  --pairing-layer 35 --optimal-layer 25 \
  --per-class 100 --local-k 32 --top-n 20 \
  --output-dir runs/lds
```

