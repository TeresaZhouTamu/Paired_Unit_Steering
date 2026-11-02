#!/usr/bin/env python3
"""Optimize target proteins with a saved steering vector and record every step.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from module.esmc_model import ESMCModel
import torch.nn.functional as F
from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN
import torch.nn.functional as F


# 1. Load a saved vector and the target proteins.
def load_vectors(path: Path) -> tuple[np.ndarray, dict]:
    """Load the vectors and provenance written by extract_steering_vectors.py."""
    with np.load(path, allow_pickle=False) as saved:
        vectors = np.asarray(saved["vectors"], dtype=np.float32)
        metadata = json.loads(str(saved["metadata"].item()))
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Vector file needs finite [layer, hidden] vectors")
    return vectors, metadata


def read_targets(path: Path, max_targets: int | None) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"sample_id": str})
    if "sequence" not in data:
        raise ValueError("Target CSV needs a sequence column")
    if data.sequence.isna().any():
        raise ValueError("Target sequences must not be empty")
    data.sequence = data.sequence.astype(str).str.strip().str.upper()
    if not data.sequence.map(lambda s: bool(s) and s.isascii() and s.isalpha()).all():
        raise ValueError("Target sequences must contain letters only")
    if "sample_id" not in data:
        data["sample_id"] = [f"target_{i:04d}" for i in range(len(data))]
    if data.sample_id.isna().any() or data.sample_id.duplicated().any():
        raise ValueError("Target sample_id values must be present and unique")
    if max_targets is not None:
        data = data.head(max_targets)
    if data.empty:
        raise ValueError("Target CSV has no sequences")
    return data


# 2. Select mutation sites and decode the next sequence at each round.
def decode_residues(model: ESMCModel, tokens) -> str:
    """Decode BOS + residue tokens + EOS with the loaded ESMC tokenizer."""
    token_ids = tokens.detach().cpu().tolist()
    if len(token_ids) < 2:
        raise ValueError("Expected BOS and EOS around residue tokens")
    residues = [
        model.tokenizer.decode([int(token_id)]).replace(" ", "")
        for token_id in token_ids[1:-1]
    ]
    if any(len(residue) != 1 for residue in residues):
        raise ValueError("Cannot decode a residue token as one amino acid")
    return "".join(residues)


def sample_top_p(probabilities, top_p: float, torch):
    """Nucleus sampling; greedy decoding is used when temperature is zero."""
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sample = torch.multinomial(sorted_probs.reshape(-1, sorted_probs.shape[-1]), 1)
    sample = sample.reshape(sorted_probs.shape[:-1])
    return sorted_indices.gather(-1, sample.unsqueeze(-1)).squeeze(-1)


def predict_tokens(model: ESMCModel, masked, original, steering, temperature, top_p):
    """Decode new amino acids while excluding the original amino acid."""
    torch = model.torch

    with torch.no_grad():
        output = model.model.steering_forward(
            masked.unsqueeze(0),
            sequence_id=torch.ones_like(masked).unsqueeze(0),
            steering_vectors=steering,
        )
    logits = output.sequence_logits[0, :, 4:24]
    original_aa = F.one_hot(original, num_classes=output.sequence_logits.shape[-1])[:, 4:24]
    logits = logits.masked_fill(original_aa.bool(), -1e8)
    if temperature == 0:
        prediction = logits.argmax(dim=-1)
    else:
        prediction = sample_top_p((logits / temperature).softmax(dim=-1), top_p, torch)
    return prediction + 4


def optimize_sequence(model: ESMCModel, sequence: str, vectors: np.ndarray, args):
    """Return endpoint, mutation log, and the sequence after every round."""
    torch = model.torc

    current = model.encode_sequence(sequence)
    steering = torch.from_numpy(vectors).to(model.device) * args.alpha
    scoring_vector = torch.from_numpy(vectors[args.optimal_layer]).to(model.device)
    previously_selected: set[int] = set()
    mutations: list[dict] = []
    steps: list[dict] = []

    for round_number in range(1, args.rounds + 1):
        before = decode_residues(model, current)
        residue_features = model.residue_representations(current)[args.optimal_layer]
        if len(residue_features) != len(current) - 2:
            raise ValueError("Expected one scoring vector per sequence residue")
        if args.site_indexing == "legacy":
            residue_features = residue_features[1:-1]
        if len(residue_features) == 0:
            break

        cosine = F.cosine_similarity(residue_features, scoring_vector[None], dim=-1)
        ranked_indices = np.argsort(cosine.cpu().numpy())  # Ascending, as in paired.ipynb.
        sites = []
        for idx in ranked_indices.tolist():
            if args.allow_repeat_sites or idx not in previously_selected:
                sites.append(idx)
                if len(sites) >= args.sites_per_round:
                    break
        if not sites:
            break
        if not args.allow_repeat_sites:
            previously_selected.update(sites)

        token_positions = torch.tensor(sites, dtype=torch.long, device=model.device) + 1
        old_tokens = current[token_positions].clone()
        masked = current.clone()
        masked[token_positions] = SEQUENCE_MASK_TOKEN
        predicted = predict_tokens(
            model, masked, current, steering, args.temperature, args.top_p
        )
        new_tokens = predicted[token_positions]
        current = current.clone()
        current[token_positions] = new_tokens

        for site, token_pos, old, new in zip(
            sites, token_positions.tolist(), old_tokens.tolist(), new_tokens.tolist()
        ):
            mutations.append({
                "round": round_number,
                "site_score_index": site,
                "token_position": token_pos,
                "residue_position_1based": token_pos,
                "old": model.tokenizer.decode([int(old)]).replace(" ", ""),
                "new": model.tokenizer.decode([int(new)]).replace(" ", ""),
                "site_cosine": float(cosine[site]),
            })
        steps.append({
            "step": round_number,
            "before_sequence": before,
            "after_sequence": decode_residues(model, current),
        })

    return decode_residues(model, current), mutations, steps


# 3. Save endpoints and the complete sequence trajectory.
def run(args) -> None:
    if min(args.rounds, args.sites_per_round) < 1 or args.alpha < 0:
        raise ValueError("Rounds and sites must be positive; alpha must be nonnegative")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("Temperature must be nonnegative and top-p in (0,1]")
    if args.max_targets is not None and args.max_targets < 1:
        raise ValueError("max-targets must be positive")

    vectors, metadata = load_vectors(args.vectors)
    model_name = args.model or str(metadata["model"])
    if model_name != metadata["model"]:
        raise ValueError("Model name differs from the vector-construction model")
    if not 0 <= args.optimal_layer < len(vectors):
        raise ValueError("--optimal-layer is outside the vector's layer range")
    targets = read_targets(args.targets, args.max_targets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_sequences = args.output_dir / "optimized_sequences.csv"
    output_steps = args.output_dir / "trajectory_steps.csv"
    output_config = args.output_dir / "run_config.json"
    if not args.overwrite and any(
        path.exists() for path in (output_sequences, output_steps, output_config)
    ):
        raise FileExistsError("Output already exists; pass --overwrite to replace it")

    random.seed(args.seed)
    np.random.seed(args.seed)
    model = ESMCModel(model_name, args.device)
    model.torch.manual_seed(args.seed)
    if model.torch.cuda.is_available():
        model.torch.cuda.manual_seed_all(args.seed)
    model.enable_steering(args.steering_module)
    if model.mean_representations(targets.iloc[0].sequence).shape != vectors.shape:
        raise ValueError("The saved vector shape does not match this ESMC model")

    method = str(metadata.get("method", "Paired-Unit"))
    sequences, trajectory_steps = [], []
    for row in targets.itertuples(index=False):
        endpoint, mutation_log, steps = optimize_sequence(model, row.sequence, vectors, args)
        sequences.append({
            "sample_id": row.sample_id, "method": method,
            "original_sequence": row.sequence, "sequence": endpoint,
            "trajectory_path": json.dumps([row.sequence, *[step["after_sequence"] for step in steps]]),
            "num_mutations": sum(m["old"] != m["new"] for m in mutation_log),
            "mutation_log": json.dumps(mutation_log),
        })
        for step in steps:
            trajectory_steps.append({
                "sample_id": row.sample_id, "method": method,
                "original_sequence": row.sequence, **step,
            })

    pd.DataFrame(sequences).to_csv(output_sequences, index=False)
    pd.DataFrame(
        trajectory_steps,
        columns=["sample_id", "method", "original_sequence", "step",
                 "before_sequence", "after_sequence"],
    ).to_csv(output_steps, index=False)
    output_config.write_text(json.dumps({
        "model": model_name, "vectors": str(args.vectors), "method": method,
        "rounds": args.rounds, "sites_per_round": args.sites_per_round,
        "optimal_layer": args.optimal_layer, "alpha": args.alpha,
        "site_indexing": args.site_indexing, "temperature": args.temperature,
        "top_p": args.top_p, "seed": args.seed,
        "steering_module": args.steering_module,
    }, indent=2) + "\n")
    print(f"Saved {len(sequences)} endpoints and {len(trajectory_steps)} steps to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, help="CSV: sequence[,sample_id]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimal-layer", "--optimal_layer", dest="optimal_layer",
                        type=int, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--sites-per-round", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--site-indexing", choices=("legacy", "standard"), default="legacy"
    )
    parser.add_argument("--allow-repeat-sites", action="store_true")
    parser.add_argument("--steering-module", default="module.steerable_esmc")
    parser.add_argument("--model", help="Defaults to the model recorded in the vector file")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
