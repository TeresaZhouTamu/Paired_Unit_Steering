#!/usr/bin/env python3
"""Construct and save Paired-Unit steering vectors from labeled proteins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from module.esmc_model import ESMCModel

# Method named "Paired-Raw", "Matched-Unit" and "Random-Unit" can be used as ablations.
EPS = 1e-8
METHODS = ("ASPO", "Paired-Unit", "Paired-Raw", "Matched-Unit", "Random-Unit")


# 1. Load the positive and negative construction proteins.
def read_construction_data(path: Path, positive_label: str, negative_label: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"label": str})
    if not {"sequence", "label"}.issubset(data.columns):
        raise ValueError("Construction CSV needs sequence and label columns")
    data = data[["sequence", "label"]].copy()
    if data.sequence.isna().any():
        raise ValueError("Construction sequences must not be empty")
    data.sequence = data.sequence.astype(str).str.strip().str.upper()
    data.label = data.label.astype(str).str.strip()
    if not data.sequence.map(lambda s: bool(s) and s.isascii() and s.isalpha()).all():
        raise ValueError("Construction sequences must contain letters only")
    if positive_label == negative_label or set(data.label) - {positive_label, negative_label}:
        raise ValueError("Labels must match --positive-label and --negative-label")
    if data.groupby("sequence").label.nunique().gt(1).any():
        raise ValueError("A construction sequence has conflicting labels")
    data = data.drop_duplicates("sequence").reset_index(drop=True)
    if min((data.label == positive_label).sum(), (data.label == negative_label).sum()) < 1:
        raise ValueError("At least one sequence of each label is required")
    return data


# 2. Match comparable backgrounds and aggregate their property directions.
def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)


def match_backgrounds(
    positive: np.ndarray, negative: np.ndarray, pairing_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find one-to-one positive/negative matches by layer cosine similarity."""
    similarity = unit_rows(positive[:, pairing_layer]) @ unit_rows(negative[:, pairing_layer]).T
    positive_idx, negative_idx = linear_sum_assignment(-similarity)
    return positive_idx, negative_idx, similarity[positive_idx, negative_idx]


def construct_vectors(
    positive: np.ndarray,
    negative: np.ndarray,
    method: str,
    pairing_layer: int,
    temperature: float = 0.10,
    keep_fraction: float = 1.0,
    random_pair_seed: int = 17,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    PU matches pairs at ``pairing_layer``, normalizes each pair displacement
    at each layer, weights pairs by matching similarity, then matches the
    centroid vector norm layer by layer.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown construction method: {method}")
    pos = np.asarray(positive, dtype=np.float32)
    neg = np.asarray(negative, dtype=np.float32)
    if pos.ndim != 3 or neg.ndim != 3 or pos.shape[1:] != neg.shape[1:]:
        raise ValueError("Representations need compatible [sequence, layer, hidden] shapes")
    if min(len(pos), len(neg)) < 1 or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("Both classes need finite representations")
    if not 0 <= pairing_layer < pos.shape[1]:
        raise ValueError("--pairing-layer is outside the model's layer range")
    if temperature <= 0 or not 0 < keep_fraction <= 1:
        raise ValueError("Temperature must be positive; keep fraction must be in (0,1]")

    aspo = pos.mean(axis=0) - neg.mean(axis=0)
    details: dict[str, object] = {
        "method": method, "positive_count": len(pos), "negative_count": len(neg),
    }
    if method == "ASPO":
        return aspo.astype(np.float32), details

    if method == "Random-Unit":
        rng = np.random.default_rng(random_pair_seed)
        n_pairs = min(len(pos), len(neg))
        if len(pos) <= len(neg):
            pos_idx = np.arange(n_pairs)
            neg_idx = rng.permutation(rng.choice(len(neg), n_pairs, replace=False))
        else:
            pos_idx = rng.permutation(rng.choice(len(pos), n_pairs, replace=False))
            neg_idx = np.arange(n_pairs)
        pair_similarity = np.sum(
            unit_rows(pos[pos_idx, pairing_layer]) * unit_rows(neg[neg_idx, pairing_layer]),
            axis=1,
        )
        details["random_pair_seed"] = random_pair_seed
    else:
        pos_idx, neg_idx, pair_similarity = match_backgrounds(pos, neg, pairing_layer)
        n_keep = max(1, round(keep_fraction * len(pos_idx)))
        keep = np.argsort(-pair_similarity, kind="stable")[:n_keep]
        pos_idx, neg_idx = pos_idx[keep], neg_idx[keep]
        pair_similarity = pair_similarity[keep]

    differences = pos[pos_idx] - neg[neg_idx]  # [pair, layer, hidden]
    if method in ("Paired-Unit", "Matched-Unit", "Random-Unit"):
        differences = unit_rows(differences)

    if method in ("Paired-Unit", "Paired-Raw"):
        logits = (pair_similarity - pair_similarity.max()) / temperature
        weights = np.exp(logits)
        weights /= weights.sum()
    else:
        weights = np.full(len(pos_idx), 1.0 / len(pos_idx), dtype=np.float32)

    weighted = np.einsum("p,pld->ld", weights, differences)
    weighted_norm = np.linalg.norm(weighted, axis=-1, keepdims=True)
    aspo_norm = np.linalg.norm(aspo, axis=-1, keepdims=True)
    if (weighted_norm <= EPS).any() or (aspo_norm <= EPS).any():
        raise ValueError("A layer has a zero-length direction; inspect construction data")
    vectors = (weighted / weighted_norm * aspo_norm).astype(np.float32)
    details.update({
        "matched_count": len(pos_idx),
        "mean_pair_cosine": float(pair_similarity.mean()),
        "effective_pairs": float(1.0 / np.sum(weights ** 2)),
        "pairing_layer": pairing_layer,
        "weight_temperature": temperature,
        "keep_fraction": keep_fraction,
        "norm_matched_to_aspo": True,
    })
    return vectors, details


# 3. Save a reusable vector with its construction settings.
def save_vectors(path: Path, vectors: np.ndarray, metadata: dict, overwrite: bool = False) -> None:
    if path.suffix != ".npz":
        raise ValueError("Vector output must have a .npz extension")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vectors=vectors, metadata=json.dumps(metadata, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction", type=Path, required=True, help="CSV: sequence,label")
    parser.add_argument("--output", type=Path, required=True, help="Saved .npz vector")
    parser.add_argument("--pairing-layer", type=int, required=True, help="sequence matching layer")
    parser.add_argument("--method", choices=METHODS, default="Paired-Unit")
    parser.add_argument("--positive-label", default="1")
    parser.add_argument("--negative-label", default="0")
    parser.add_argument("--weight-temperature", type=float, default=0.10)
    parser.add_argument("--keep-fraction", type=float, default=1.0)
    parser.add_argument("--random-pair-seed", type=int, default=17)
    parser.add_argument("--model", default="esmc_600m")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = read_construction_data(args.construction, args.positive_label, args.negative_label)
    encoder = ESMCModel(args.model, args.device)
    embeddings = np.stack([encoder.mean_representations(seq) for seq in data.sequence])
    vectors, details = construct_vectors(
        embeddings[data.label == args.positive_label],
        embeddings[data.label == args.negative_label],
        args.method, args.pairing_layer, args.weight_temperature,
        args.keep_fraction, args.random_pair_seed,
    )
    details.update({
        "format": "steering_vectors_v1", "model": args.model,
        "construction_sha256": hashlib.sha256(args.construction.read_bytes()).hexdigest(),
        "layers": vectors.shape[0], "hidden_size": vectors.shape[1],
        "positive_label": args.positive_label, "negative_label": args.negative_label,
    })
    save_vectors(args.output, vectors, details, overwrite=args.overwrite)
    print(json.dumps({"vectors": str(args.output), **details}, indent=2))


if __name__ == "__main__":
    main()
